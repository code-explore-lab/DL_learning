import time

import cupy as cp
import numpy as np

# ========== 基础版：相邻两元素相加，s 从 1 倍增；用 tid % (2s)==0 选活跃线程 ==========
kernel_basic = r"""
extern "C"
__global__ void sum_shared_basic(const float* x, float* partial, int n)
{
## Block内共享内存
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid; //全局索引
    sdata[tid] = (i < n) ? x[i] : 0.f;
    __syncthreads(); //block内线程屏障同步

    // log2(BLOCK)=8 步，相邻两个元素相加
    for (int s = 1; s < blockDim.x; s <<= 1) {
        if (tid % (2 * s) == 0) {          // 每个 warp 内部一半/四分之一... lane 空转
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = sdata[0];
}
"""

# ========== 对照：顺序寻址，把取模换成 tid < s，减少分叉和取模开销 ==========
kernel_seq = r"""
extern "C"
__global__ void sum_shared_seq(const float* x, float* partial, int n)
{
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? x[i] : 0.f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {                      // 活跃线程连续，前期整 warp 全活/全不活
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = sdata[0];
}
"""

sum_basic = cp.RawKernel(kernel_basic, "sum_shared_basic")
sum_seq = cp.RawKernel(kernel_seq, "sum_shared_seq")

BLOCK = 256
N = 1_000_000
GRID = (N + BLOCK - 1) // BLOCK

rng = np.random.default_rng(0)
n_x = rng.standard_normal(N, dtype=np.float32)
d_x = cp.asarray(n_x)
ref = float(np.sum(n_x.astype(np.float64)))

d_partial = cp.empty(GRID, dtype=cp.float32)


def launch_basic():
    sum_basic((GRID, 1, 1), (BLOCK, 1, 1), (d_x, d_partial, np.int32(N)))


def launch_seq():
    sum_seq((GRID, 1, 1), (BLOCK, 1, 1), (d_x, d_partial, np.int32(N)))


def bench_gpu(fn, warmup=10, repeat=200):
    for _ in range(warmup):
        fn()
    cp.cuda.get_current_stream().synchronize()
    e0, e1 = cp.cuda.Event(), cp.cuda.Event()
    e0.record()
    for _ in range(repeat):
        fn()
    e1.record()
    cp.cuda.get_current_stream().synchronize()
    return cp.cuda.get_elapsed_time(e0, e1) / repeat


def cpu_reduce_and_check(launch):
    launch()
    p = cp.asnumpy(d_partial)
    total = float(np.sum(p))                 # CPU 再 reduce
    return total, abs(total - ref) / abs(ref), p


def bench_cpu(p, warmup=10, repeat=200):
    for _ in range(warmup):
        np.sum(p)
    t0 = time.perf_counter()
    for _ in range(repeat):
        np.sum(p)
    return (time.perf_counter() - t0) / repeat * 1000.0


# ========== 正确性 + 计时 ==========
print(f"N={N:,}, BLOCK={BLOCK}, GRID={GRID}, 每 block 规约 {BLOCK} 个元素\n")

total_b, err_b, p_b = cpu_reduce_and_check(launch_basic)
total_s, err_s, p_s = cpu_reduce_and_check(launch_seq)

t_b = bench_gpu(launch_basic)
t_s = bench_gpu(launch_seq)
t_cpu = bench_cpu(p_b)

print(f"{'方式':<26}{'kernel(ms)':>12}{'CPU reduce(ms)':>16}{'相对误差':>12}")
print("-" * 66)
print(f"{'基础版(相邻+取模)':<24}{t_b:>12.4f}{t_cpu:>16.4f}{err_b:>12.2e}")
print(f"{'对照:顺序寻址':<25}{t_s:>12.4f}{t_cpu:>16.4f}{err_s:>12.2e}")
print("-" * 66)
print(f"顺序寻址 / 基础版: {t_b / t_s:.2f}x")


def bench_cupy_sum(warmup=10, repeat=200):
    for _ in range(warmup):
        d_x.sum()
    cp.cuda.get_current_stream().synchronize()
    e0, e1 = cp.cuda.Event(), cp.cuda.Event()
    e0.record()
    for _ in range(repeat):
        d_x.sum()
    e1.record()
    cp.cuda.get_current_stream().synchronize()
    return cp.cuda.get_elapsed_time(e0, e1) / repeat


t_ref = bench_cupy_sum()
print(f"cupy d_x.sum(): {t_ref * 1000:.2f} us"
      f"（基础版 {t_b / t_ref:.1f}x，顺序寻址 {t_s / t_ref:.1f}x）")

# ========== 显示 block 内归约的中间结果，确认相邻相加 ==========
print("\n16 个元素、4 步归约的中间过程（sdata[0..15]，用 Python 复刻基础版算法）")
step_demo = np.zeros(16, dtype=np.float32)
step_demo[:] = np.arange(1, 17, dtype=np.float32)
print("初始 :", step_demo[:16])
for s in (1, 2, 4, 8):
    for tid in range(0, 16, 2 * s):
        step_demo[tid] += step_demo[tid + s]
    print(f"s={s:<2}:", step_demo[:16])
print(f"总和: {step_demo[0]}")
