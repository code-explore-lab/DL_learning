import time

import cupy as cp
import numpy as np

# ========== Naive：block 内每个线程把局部和 atomicAdd 到全局 partial[blockIdx.x] ==========
kernel_code = r"""
extern "C"
__global__ void sum_naive(const float* x, float* partial, int n)
{
    int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float s = 0.f;
    for (; i < n; i += stride) {
        s += x[i];
    }
    //`partial[blockIdx.x]` 是这个 block 对应的全局内存位置；
    //  **同一个 block 里面所有 256 个线程，都往同一个全局地址做原子加**。
    atomicAdd(&partial[blockIdx.x], s);   // 直接打全局内存，既没有 shared 也没有 warp shuffle
}
"""
sum_naive = cp.RawKernel(kernel_code, "sum_naive")

BLOCK = 256
N = 1_000_000

rng = np.random.default_rng(0)
n_x = rng.standard_normal(N, dtype=np.float32)
d_x = cp.asarray(n_x)
ref = float(np.sum(n_x.astype(np.float64)))     # CPU 双精度参考值


_partial_pool = {}


def run_naive(grid_blocks):
    """完整流程：清零 partial -> GPU 算 block 部分和 -> CPU reduce"""
    d_partial = _partial_pool.get(grid_blocks)
    if d_partial is None:
        d_partial = cp.zeros(grid_blocks, dtype=cp.float32)
        _partial_pool[grid_blocks] = d_partial
    else:
        d_partial.fill(0.0)          # atomic 累加前必须清零，算在耗时里
    sum_naive((grid_blocks, 1, 1), (BLOCK, 1, 1), (d_x, d_partial, np.int32(N)))
    return d_partial


def bench_kernel(grid_blocks, warmup=10, repeat=100):
    """只计 GPU 部分（含每次必须的 partial 清零）"""
    for _ in range(warmup):
        run_naive(grid_blocks)
    cp.cuda.get_current_stream().synchronize()
    e0, e1 = cp.cuda.Event(), cp.cuda.Event()
    e0.record()
    for _ in range(repeat):
        run_naive(grid_blocks)
    e1.record()
    cp.cuda.get_current_stream().synchronize()
    return cp.cuda.get_elapsed_time(e0, e1) / repeat


def bench_cpu_reduce(grid_blocks, warmup=10, repeat=100):
    p = cp.asnumpy(run_naive(grid_blocks))
    for _ in range(warmup):
        np.sum(p)
    t0 = time.perf_counter()
    for _ in range(repeat):
        np.sum(p)
    return (time.perf_counter() - t0) / repeat * 1000.0


# ========== 不同 block 数：越多 -> 全局 atomic 越多 ==========
floor_us = N * 4 / 300e9 * 1e6
print(f"N={N:,}, BLOCK={BLOCK}, 只读数据 {N * 4 / 1e6:.1f} MB"
      f"（带宽下限约 {floor_us:.1f} us @300GB/s）\n")
print(f"{'GRID':>6}{'每线程元素':>12}{'atomic次数':>11}"
      f"{'kernel(ms)':>12}{'CPU reduce(ms)':>16}{'相对误差':>12}")

results = {}
for grid_blocks in (32, 64, 128, 256, 512, 1024, 2048, 4096):
    d_partial = run_naive(grid_blocks)
    total = float(np.sum(cp.asnumpy(d_partial)))
    rel_err = abs(total - ref) / abs(ref)
    t_gpu = bench_kernel(grid_blocks)
    t_cpu = bench_cpu_reduce(grid_blocks)
    results[grid_blocks] = t_gpu
    per_thread = (N + grid_blocks * BLOCK - 1) // (grid_blocks * BLOCK)
    print(f"{grid_blocks:>6}{per_thread:>12}{grid_blocks * BLOCK:>11}"
          f"{t_gpu:>12.4f}{t_cpu:>16.4f}{rel_err:>12.2e}")

# ========== 对照：cupy 自己的归约 ==========
def bench_cupy_sum(warmup=10, repeat=100):
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
best_grid = min(results, key=results.get)
print(f"\ncupy d_x.sum()（优化归约）: {t_ref * 1000:.2f} us")
print(f"naive 最好一档 GRID={best_grid}: {results[best_grid] * 1000:.2f} us"
      f"（比 cupy 慢 {results[best_grid] / t_ref:.1f}x）")
print(f"naive 最差一档 GRID={max(results, key=results.get)}: "
      f"{max(results.values()) * 1000:.2f} us"
      f"（比 cupy 慢 {max(results.values()) / t_ref:.1f}x）")
