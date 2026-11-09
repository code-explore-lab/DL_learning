import cupy as cp
import numpy as np

# ========== 标准精度：expf ==========
kernel_std = r"""
extern "C"
__global__ void sigmoid_std(const float* x, float* out, int n)
{
    int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += stride) {
        out[i] = 1.f / (1.f + expf(-x[i]));
    }
}
"""

# ========== 快精度：__expf（GPU 硬件近似指令） ==========
kernel_fast = r"""
extern "C"
__global__ void sigmoid_fast(const float* x, float* out, int n)
{
    int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += stride) {
        out[i] = 1.f / (1.f + __expf(-x[i]));
    }
}
"""

# ========== 计算密集版：每个元素连续做 8 次 sigmoid，把 exp 吞吐暴露出来 ==========
kernel_std_chain = r"""
extern "C"
__global__ void sigmoid_std_chain(const float* x, float* out, int n)
{
    int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += stride) {
        float v = x[i];
        #pragma unroll
        for (int k = 0; k < 64; ++k) v = 1.f / (1.f + expf(-v));
        out[i] = v;
    }
}
"""
kernel_fast_chain = r"""
extern "C"
__global__ void sigmoid_fast_chain(const float* x, float* out, int n)
{
    int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += stride) {
        float v = x[i];
        #pragma unroll
        for (int k = 0; k < 64; ++k) v = 1.f / (1.f + __expf(-v));
        out[i] = v;
    }
}
"""

sigmoid_std = cp.RawKernel(kernel_std, "sigmoid_std")
sigmoid_fast = cp.RawKernel(kernel_fast, "sigmoid_fast")
sigmoid_std_chain = cp.RawKernel(kernel_std_chain, "sigmoid_std_chain")
sigmoid_fast_chain = cp.RawKernel(kernel_fast_chain, "sigmoid_fast_chain")

BLOCK = 256
N = 20_000_000
GRID = ((N + BLOCK - 1) // BLOCK, 1, 1)

d_x = cp.random.randn(N, dtype=cp.float32)
d_out_std = cp.empty_like(d_x)
d_out_fast = cp.empty_like(d_x)


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
    return cp.cuda.get_elapsed_time(e0, e1) / repeat


# ========== 计时：两边都写入预分配输出，只换 exp 实现 ==========
t_std = bench(lambda: sigmoid_std(GRID, (BLOCK, 1, 1), (d_x, d_out_std, np.int32(N))))
t_fast = bench(lambda: sigmoid_fast(GRID, (BLOCK, 1, 1), (d_x, d_out_fast, np.int32(N))))

print("=== 访存密集（每个元素 1 次 sigmoid，被内存带宽掩盖）===")
print(f"标准 expf   : {t_std:.4f} ms")
print(f"快精度 __expf: {t_fast:.4f} ms")
print(f"加速比: {t_std / t_fast:.2f}")

d_chain_std = cp.empty_like(d_x)
d_chain_fast = cp.empty_like(d_x)
t_std_c = bench(lambda: sigmoid_std_chain(GRID, (BLOCK, 1, 1), (d_x, d_chain_std, np.int32(N))),
                warmup=3, repeat=10)
t_fast_c = bench(lambda: sigmoid_fast_chain(GRID, (BLOCK, 1, 1), (d_x, d_chain_fast, np.int32(N))),
                 warmup=3, repeat=10)

print("=== 计算密集（每个元素 64 次 sigmoid，exp 吞吐主导）===")
print(f"标准 expf   : {t_std_c:.4f} ms")
print(f"快精度 __expf: {t_fast_c:.4f} ms")
print(f"加速比: {t_std_c / t_fast_c:.2f}")

# 单次 sigmoid 的结果重新算一遍，避免被 chain 覆盖
sigmoid_std(GRID, (BLOCK, 1, 1), (d_x, d_out_std, np.int32(N)))
sigmoid_fast(GRID, (BLOCK, 1, 1), (d_x, d_out_fast, np.int32(N)))

# ========== 正确性：两边分别对照 cupy 参考实现 ==========
d_gold = 1.0 / (1.0 + cp.exp(-d_x))
cp.testing.assert_allclose(d_out_std, d_gold, rtol=1e-6, atol=1e-7)
cp.testing.assert_allclose(d_out_fast, d_gold, rtol=1e-5, atol=1e-6)
print("两边都通过 cupy 参考校验")

# ========== 精度差：__expf 到底差多少 ==========
diff = cp.abs(d_out_fast - d_out_std)
rel = diff / cp.maximum(cp.abs(d_out_std), 1e-30)
print(f"快精度 vs 标准精度: max_abs={float(diff.max()):.3e}  max_rel={float(rel.max()):.3e}")

# ========== 边界值 ==========
edge = cp.array([-np.inf, np.inf, 0.0, -100.0, 100.0, np.nan], dtype=cp.float32)
e_std = cp.empty_like(edge)
e_fast = cp.empty_like(edge)
sigmoid_std((1, 1, 1), (BLOCK, 1, 1), (edge, e_std, np.int32(edge.size)))
sigmoid_fast((1, 1, 1), (BLOCK, 1, 1), (edge, e_fast, np.int32(edge.size)))
print("边界输入  :", edge.get())
print("标准 expf :", e_std.get())
print("快精度    :", e_fast.get())
