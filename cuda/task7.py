import cupy as cp
import numpy as np

kernel_code = r"""
extern "C"
__global__ void axpy_kernel(float a, const float* x, float* y, int n)
{
    int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}
"""
axpy_kernel = cp.RawKernel(kernel_code, "axpy_kernel")

BLOCK = 256
N = 20_000_000
a_val = 2.5

# 基准数据（只读，不变）
d_x = cp.random.randn(N, dtype=cp.float32)
d_y_base = cp.random.randn(N, dtype=cp.float32)

# 预热 + 计时
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


# ========== 手写kernel计时：每次跑之前先重置y ==========
def run_my_axpy():
    d_y = d_y_base.copy()
    axpy_kernel((4096,1,1), (BLOCK,1,1), (cp.float32(a_val), d_x, d_y, np.int32(N)))

t_my = bench(run_my_axpy)

# ========== cupy原生计时：每次跑之前重置y ==========
def run_cupy_axpy():
    d_y_ref = d_y_base.copy()
    d_y_ref[:] = a_val * d_x + d_y_ref

t_cupy = bench(run_cupy_axpy)

print(f"手写axpy: {t_my:.4f} ms")
print(f"cupy原生axpy: {t_cupy:.4f} ms")
print(f"速度比值: {t_cupy / t_my:.2f}")

# ========== 正确性校验：单独跑一次，不参与bench循环 ==========
d_y_test = d_y_base.copy()
axpy_kernel((4096,1,1), (BLOCK,1,1), (cp.float32(a_val), d_x, d_y_test, np.int32(N)))

d_y_gold = d_y_base.copy()
d_y_gold[:] = a_val * d_x + d_y_gold

cp.testing.assert_allclose(d_y_test, d_y_gold, atol=1e-5)
print("结果校验通过")
