import cupy as cp
import numpy as np

kernel_code = r"""
extern "C" __global__ void my_add(float* a, float* b, float* out, int n)
{
    // stride 同时在跑的线程总数
    int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}
"""
my_add_kernel = cp.RawKernel(kernel_code, "my_add")

BLOCK = 256
N = 20_000_000

# 数据
d_a = cp.random.randn(N, dtype=cp.float32)
d_b = cp.random.randn(N, dtype=cp.float32)
d_out = cp.empty_like(d_a)

# 预热 + 计时（复用Task5 bench）
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

t_my = bench(lambda : my_add_kernel((4096,1,1), (BLOCK,1,1), (d_a, d_b, d_out, np.int32(N))))
t_cupy = bench(lambda : d_a + d_b)

print(f"手写my_add: {t_my:.4f} ms")
print(f"cupy原生+: {t_cupy:.4f} ms")
print(f"速度比值: {t_cupy / t_my:.2f}")

# 正确性校验
cp.testing.assert_allclose(d_out, d_a + d_b, atol=1e-5)
print("✅ 结果校验通过")
