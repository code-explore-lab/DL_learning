import cupy as cp
import numpy as np

kernel_code = r"""
extern "C"
__global__ void relu_kernel(const float* x, float* out, int n)
{
    // stride = 同时在跑的线程总数
    int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += stride) {
        // GPU 上没有 std::max，用 CUDA 数学库的 fmaxf
        out[i] = fmaxf(0.f, x[i]);
    }
}
"""
relu_kernel = cp.RawKernel(kernel_code, "relu_kernel")

BLOCK = 256
N = 20_000_000
# 按数据量算 grid，而不是写死 4096
GRID = ((N + BLOCK - 1) // BLOCK, 1, 1)

d_x = cp.random.randn(N, dtype=cp.float32)
d_out = cp.empty_like(d_x)


# 预热 + 计时（复用前几个 task 的 bench）
def bench(fn, warmup=5, repeat=20):
    for _ in range(warmup):
        fn()
    cp.cuda.get_current_stream().synchronize()
    e0 = cp.cuda.Event()
    e1 = cp.cuda.Event()
    e0.record()
    for _ in range(repeat):
        fn()
    e1.record()
    cp.cuda.get_current_stream().synchronize()
    t_ms = cp.cuda.get_elapsed_time(e0, e1)
    return t_ms / repeat


# ========== 计时：两边都写入预分配的 d_out，公平对比 ==========
t_my = bench(lambda: relu_kernel(GRID, (BLOCK, 1, 1), (d_x, d_out, np.int32(N))))
t_cupy = bench(lambda: cp.maximum(d_x, cp.float32(0), out=d_out))

print(f"手写relu: {t_my:.4f} ms")
print(f"cupy原生maximum: {t_cupy:.4f} ms")
print(f"速度比值: {t_cupy / t_my:.2f}")

# ========== 正确性校验：单独跑一次，不参与 bench 循环 ==========
relu_kernel(GRID, (BLOCK, 1, 1), (d_x, d_out, np.int32(N)))
d_gold = cp.maximum(d_x, cp.float32(0))
cp.testing.assert_allclose(d_out, d_gold, atol=1e-5)
print("结果校验通过")

# ========== 边界值校验：NaN / -0.0 / ±inf ==========
edge = cp.array([np.nan, -0.0, 0.0, -np.inf, np.inf, -1e-30, 1e-30], dtype=cp.float32)
edge_out = cp.empty_like(edge)
relu_kernel((1, 1, 1), (BLOCK, 1, 1), (edge, edge_out, np.int32(edge.size)))
print("边界输入:", edge.get())
print("边界输出:", edge_out.get())
# fmaxf 的语义：有一边是 NaN 就返回另一边，所以 fmaxf(0, NaN) == 0
