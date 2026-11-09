import cupy as cp
import numpy as np

kernel_code = r"""
extern "C"
__global__ void scale_kernel(float a, const float* x, float* out, int n)
{
    int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += stride) {
        out[i] = a * x[i];          // 标量 a 对所有线程广播
    }
}
"""

# 延伸：元素级 stride 版本，i = tid * step，只处理下标为 step 倍数的元素
kernel_stride_code = r"""
extern "C"
__global__ void scale_stride_kernel(float a, const float* x, float* out, int n, int step)
{
    int grid = gridDim.x * blockDim.x;
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * step;
    for (; i < n; i += grid * step) {
        out[i] = a * x[i];
    }
}
"""

scale_kernel = cp.RawKernel(kernel_code, "scale_kernel")
scale_stride_kernel = cp.RawKernel(kernel_stride_code, "scale_stride_kernel")

BLOCK = 256
N = 20_000_000
a_val = 2.5

d_x = cp.random.randn(N, dtype=cp.float32)
d_out = cp.empty_like(d_x)


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


def grid_for(step):
    return (((N + step - 1) // step) + BLOCK - 1) // BLOCK, 1, 1


# ========== 连续版：out = a * x ==========
t_my = bench(lambda: scale_kernel(grid_for(1), (BLOCK, 1, 1),
                                  (cp.float32(a_val), d_x, d_out, np.int32(N))))
t_cupy = bench(lambda: cp.multiply(d_x, cp.float32(a_val), out=d_out))

print("=== 连续 scale（N=20M）===")
print(f"手写kernel : {t_my:.4f} ms")
print(f"cupy乘法   : {t_cupy:.4f} ms")
print(f"速度比值   : {t_cupy / t_my:.2f}")

scale_kernel(grid_for(1), (BLOCK, 1, 1), (cp.float32(a_val), d_x, d_out, np.int32(N)))
cp.testing.assert_allclose(d_out, a_val * d_x, rtol=1e-5, atol=1e-6)
print("连续版校验通过")

# ========== stride 版：只处理偶数下标，奇数下标保持哨兵值 ==========
STEP = 2
d_out_s = cp.full(N, -1.0, dtype=cp.float32)      # 哨兵，用来验证奇数位没被碰
scale_stride_kernel(grid_for(STEP), (BLOCK, 1, 1),
                    (cp.float32(a_val), d_x, d_out_s, np.int32(N), np.int32(STEP)))
cp.testing.assert_allclose(d_out_s[::STEP], a_val * d_x[::STEP], rtol=1e-5, atol=1e-6)
assert bool(cp.all(d_out_s[1::STEP] == -1.0)), "奇数下标被意外改写了"
print(f"stride={STEP} 校验通过：偶数位正确，奇数位仍是哨兵 -1")

# ========== 不同 step 的耗时 / 有效吞吐 ==========
print("\n=== 元素 stride 对带宽效率的影响 ===")
print(f"{'step':>4}  {'耗时(ms)':>9}  {'处理元素':>9}  {'有效带宽(GB/s)':>15}")
for step in (1, 2, 4, 8):
    t = bench(lambda s=step: scale_stride_kernel(
        grid_for(s), (BLOCK, 1, 1),
        (cp.float32(a_val), d_x, d_out, np.int32(N), np.int32(s))))
    touched = (N + step - 1) // step
    useful_gbs = touched * 8 / (t * 1e-3) / 1e9      # 每个元素读 4B + 写 4B
    print(f"{step:>4}  {t:>9.4f}  {touched:>9}  {useful_gbs:>15.1f}")

# 对照：cupy 的切片写法（会先 gather 再 scatter，多几个 kernel）
t_cupy_s = bench(lambda: cp.multiply(d_x[::STEP], cp.float32(a_val), out=d_out_s[::STEP]))
print(f"\ncupy 切片乘法 step={STEP}: {t_cupy_s:.4f} ms（手写: "
      f"{bench(lambda: scale_stride_kernel(grid_for(STEP), (BLOCK, 1, 1), (cp.float32(a_val), d_x, d_out, np.int32(N), np.int32(STEP)))):.4f} ms）")
