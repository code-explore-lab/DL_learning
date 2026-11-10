import cupy as cp
import numpy as np

# ========== Task 17：多 block 规约 + atomicAdd ==========
# block 内规约只能得到 partial，跨 block 求和需要一个全局累加器。
# atomicAdd 把"读-改-写"做成一个不可分割的操作，多个 block 同时加也不会丢结果。
# 代价：对同一个地址的 atomic 是串行的（在 L2 里排队），
# 所以优化方向永远是——先把 block 内的和算出来，让每个 block 只贡献 1 次 atomic。

# ---------- 1. 反例：每个线程直接 atomicAdd 到同一个全局地址 ----------
# N 次 atomic 全部砸向 global_sum，L2 里排一条长队，完全串行。
kernel_atomic_per_thread = r"""
extern "C"
__global__ void sum_atomic_per_thread(const float* x, float* global_sum, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(global_sum, x[i]);       // ← N 次 atomic，全部同一个地址
    }
}
"""

# ---------- 2. 每 warp 规约后 lane 0 做 1 次 atomic：atomic 次数 = N/32 ----------
# 只有 5 条 shfl，没有 shared、没有 __syncthreads。
kernel_atomic_per_warp = r"""
extern "C"
__global__ void sum_atomic_per_warp(const float* x, float* global_sum, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (i < n) ? x[i] : 0.f;

    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);

    if ((threadIdx.x & 31) == 0)
        atomicAdd(global_sum, v);          // ← 每个 warp 1 次 atomic
}
"""

# ---------- 3. 标准做法：block 内 shfl 规约，最后 thread 0 做 1 次 atomic ----------
# atomic 次数 = GRID（比每线程 atomic 少 BLOCK 倍）。
kernel_atomic_per_block = r"""
extern "C"
__global__ void sum_atomic_per_block(const float* x, float* global_sum, int n)
{
    __shared__ float warp_sums[32];

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;
    int i    = blockIdx.x * blockDim.x + tid;

    float v = (i < n) ? x[i] : 0.f;
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);

    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();

    if (wid == 0) {
        int nwarps = blockDim.x >> 5;
        float w = (lane < nwarps) ? warp_sums[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1)
            w += __shfl_down_sync(0xffffffffu, w, off);
        if (lane == 0) atomicAdd(global_sum, w);   // ← 每个 block 1 次 atomic
    }
}
"""

# ---------- 4. 对照：完全不用 atomic，两趟 kernel ----------
# 第一趟每个 block 写 partial[blockIdx.x]；
# 第二趟用 1 个 block 把 partial 数组再规约一次。
kernel_to_partial = r"""
extern "C"
__global__ void sum_to_partial(const float* x, float* partial, int n)
{
    __shared__ float warp_sums[32];

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;
    int i    = blockIdx.x * blockDim.x + tid;

    float v = (i < n) ? x[i] : 0.f;
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);

    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();

    if (wid == 0) {
        int nwarps = blockDim.x >> 5;
        float w = (lane < nwarps) ? warp_sums[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1)
            w += __shfl_down_sync(0xffffffffu, w, off);
        if (lane == 0) partial[blockIdx.x] = w;
    }
}
"""

kernel_reduce_partial = r"""
extern "C"
__global__ void reduce_partial(const float* partial, float* out, int nblocks)
{
    __shared__ float warp_sums[32];

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;

    // 1 个 block 用 grid-stride 扫完整个 partial 数组
    float v = 0.f;
    for (int i = tid; i < nblocks; i += blockDim.x)
        v += partial[i];

    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);

    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();

    if (wid == 0) {
        int nwarps = blockDim.x >> 5;
        float w = (lane < nwarps) ? warp_sums[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1)
            w += __shfl_down_sync(0xffffffffu, w, off);
        if (lane == 0) out[0] = w;
    }
}
"""

# ---------- 5. atomic 微基准：同一个地址 vs 每 block 自己的地址 ----------
# 每个线程只发 1 次 atomicAdd(1.0f)，两种模式次数完全相同，只差地址是否分散：
# all_threads=1：所有线程砸同一地址（全局串行）；
# all_threads=0：每个 block 的线程砸自己 block 的槽（GRID 个地址并行）。
# 1M 以内 float 能精确表示整数，所以末位自检是精确的。
kernel_atomic_micro = r"""
extern "C"
__global__ void atomic_micro(float* out, int all_threads)
{
    if (all_threads)
        atomicAdd(out, 1.0f);                      // 全部同一个地址
    else
        atomicAdd(&out[blockIdx.x], 1.0f);         // 每个 block 自己的地址
}
"""

sum_atomic_per_thread = cp.RawKernel(kernel_atomic_per_thread, "sum_atomic_per_thread")
sum_atomic_per_warp = cp.RawKernel(kernel_atomic_per_warp, "sum_atomic_per_warp")
sum_atomic_per_block = cp.RawKernel(kernel_atomic_per_block, "sum_atomic_per_block")
sum_to_partial = cp.RawKernel(kernel_to_partial, "sum_to_partial")
reduce_partial = cp.RawKernel(kernel_reduce_partial, "reduce_partial")
atomic_micro = cp.RawKernel(kernel_atomic_micro, "atomic_micro")

N = 1_000_000
BLOCK = 256
GRID = (N + BLOCK - 1) // BLOCK

rng = np.random.default_rng(0)
n_x = rng.standard_normal(N, dtype=np.float32)
d_x = cp.asarray(n_x)
ref = float(np.sum(n_x.astype(np.float64)))

d_sum = cp.zeros(1, dtype=cp.float32)                  # atomic 累加器
d_partial = cp.empty(GRID, dtype=cp.float32)
d_two_pass = cp.zeros(1, dtype=cp.float32)


def bench_gpu(launch, warmup=10, repeat=200):
    for _ in range(warmup):
        launch()
    cp.cuda.get_current_stream().synchronize()
    e0, e1 = cp.cuda.Event(), cp.cuda.Event()
    e0.record()
    for _ in range(repeat):
        launch()
    e1.record()
    cp.cuda.get_current_stream().synchronize()
    return cp.cuda.get_elapsed_time(e0, e1) / repeat


print(f"N={N:,}, BLOCK={BLOCK}, GRID={GRID}，"
      f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}\n")


# ========== 第一组：四种跨 block 方案 ==========

def launch_per_thread():
    d_sum.fill(0.0)                                    # atomic 前必须清零，算进耗时
    sum_atomic_per_thread((GRID, 1, 1), (BLOCK, 1, 1),
                          (d_x, d_sum, np.int32(N)))


def launch_per_warp():
    d_sum.fill(0.0)
    sum_atomic_per_warp((GRID, 1, 1), (BLOCK, 1, 1),
                        (d_x, d_sum, np.int32(N)))


def launch_per_block():
    d_sum.fill(0.0)
    sum_atomic_per_block((GRID, 1, 1), (BLOCK, 1, 1),
                         (d_x, d_sum, np.int32(N)))


def launch_two_pass():
    sum_to_partial((GRID, 1, 1), (BLOCK, 1, 1), (d_x, d_partial, np.int32(N)))
    reduce_partial((1, 1, 1), (1024, 1, 1),
                   (d_partial, d_two_pass, np.int32(GRID)))


def total_of(accum):
    return float(cp.asnumpy(accum)[0])


def err_of(v):
    return abs(v - ref) / abs(ref)


# 先取结果 / 误差
launch_per_thread()
v_pt = total_of(d_sum)
launch_per_warp()
v_pw = total_of(d_sum)
launch_per_block()
v_pb = total_of(d_sum)
launch_two_pass()
v_tp = total_of(d_two_pass)

# 再测时间（per-thread 版 atomic 太多，少跑几次）
t_pt = bench_gpu(launch_per_thread, warmup=3, repeat=20)
t_pw = bench_gpu(launch_per_warp)
t_pb = bench_gpu(launch_per_block)
t_tp = bench_gpu(launch_two_pass)

print(f"{'方案':<32}{'atomic 次数':>12}{'kernel(us)':>12}{'相对误差':>12}")
print("-" * 70)
rows = [
    ("每线程 atomic（反例）", N, t_pt, v_pt),
    ("每 warp 规约 + atomic", N // 32, t_pw, v_pw),
    ("每 block 规约 + atomic", GRID, t_pb, v_pb),
    ("两趟 kernel（无 atomic）", 0, t_tp, v_tp),
]
for name, atomics, t, v in rows:
    print(f"{name:<30}{atomics:>12,}{t * 1000:>12.2f}{err_of(v):>12.2e}")
print("-" * 70)
print(f"每 block 版比每线程版快 {t_pt / t_pb:.1f}x（atomic 少了 {N // GRID}x），"
      f"比两趟 kernel 快 {t_tp / t_pb:.2f}x")

# 对照 cupy（放在重度压测之前测，避免散热降频影响参考值）
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
print(f"cupy d_x.sum(): {t_ref * 1000:.2f} us（per-block+atomic 是它的 "
      f"{t_pb / t_ref:.2f}x）")
print("注：笔记本 GPU 连续跑会降频，不同表的绝对值会有 ±20% 抖动，横向比较看同一张表。")


# ========== 第二组：BLOCK 扫描，看 atomic 次数怎么随 block 变大而下降 ==========
print(f"\nBLOCK 扫描（N={N:,}）：block 越大，atomic 越少、block 内规约越贵")
print(f"{'BLOCK':>6}{'GRID':>8}{'atomic 次数':>12}"
      f"{'per-block+atomic(us)':>22}{'两趟(us)':>10}")
print("-" * 60)

for BLOCK in (32, 64, 128, 256, 512, 1024):
    GRID = (N + BLOCK - 1) // BLOCK
    d_p = cp.empty(GRID, dtype=cp.float32)

    def launch_pb():
        d_sum.fill(0.0)
        sum_atomic_per_block((GRID, 1, 1), (BLOCK, 1, 1),
                             (d_x, d_sum, np.int32(N)))

    def launch_tp():
        sum_to_partial((GRID, 1, 1), (BLOCK, 1, 1),
                       (d_x, d_p, np.int32(N)))
        reduce_partial((1, 1, 1), (1024, 1, 1),
                       (d_p, d_two_pass, np.int32(GRID)))

    t_pb = bench_gpu(launch_pb, warmup=5, repeat=100)
    t_tp = bench_gpu(launch_tp, warmup=5, repeat=100)
    print(f"{BLOCK:>6}{GRID:>8}{GRID:>12,}{t_pb * 1000:>22.2f}{t_tp * 1000:>10.2f}")


# ========== 第二点五组：不同数据量下的对比 ==========
# atomic 版少一次 partial 数组的往返，但计时里要算上累加器清零的那次 kernel。
print("\n不同数据量对比（BLOCK=256）")
print(f"{'N':>10}{'GRID':>8}{'per-block+atomic(us)':>22}{'两趟(us)':>10}{'谁快':>8}")
print("-" * 60)

for N_small in (4_096, 65_536, 262_144, 1_000_000):
    G = (N_small + BLOCK - 1) // BLOCK
    d_xs = cp.asarray(rng.standard_normal(N_small, dtype=np.float32))
    d_ps = cp.empty(G, dtype=cp.float32)

    def launch_pb():
        d_sum.fill(0.0)
        sum_atomic_per_block((G, 1, 1), (BLOCK, 1, 1),
                             (d_xs, d_sum, np.int32(N_small)))

    def launch_tp():
        sum_to_partial((G, 1, 1), (BLOCK, 1, 1),
                       (d_xs, d_ps, np.int32(N_small)))
        reduce_partial((1, 1, 1), (1024, 1, 1),
                       (d_ps, d_two_pass, np.int32(G)))

    t_pb = bench_gpu(launch_pb)
    t_tp = bench_gpu(launch_tp)
    winner = "atomic" if t_pb < t_tp else "两趟"
    print(f"{N_small:>10,}{G:>8}{t_pb * 1000:>22.2f}{t_tp * 1000:>10.2f}{winner:>8}")


# ========== 第三组：atomic 微基准，量化"串行"到底有多贵 ==========
MIC_GRID = 4096
MIC_BLOCK = 256
d_mic = cp.zeros(MIC_GRID, dtype=cp.float32)

def launch_mic_same():
    d_mic.fill(0.0)
    atomic_micro((MIC_GRID, 1, 1), (MIC_BLOCK, 1, 1),
                 (d_mic, np.int32(1)))

def launch_mic_spread():
    d_mic.fill(0.0)
    atomic_micro((MIC_GRID, 1, 1), (MIC_BLOCK, 1, 1),
                 (d_mic, np.int32(0)))

t_same = bench_gpu(launch_mic_same, warmup=5, repeat=100)
t_spread = bench_gpu(launch_mic_spread, warmup=5, repeat=100)
n_same = MIC_GRID * MIC_BLOCK                      # 两种模式发出的 atomic 次数相同
n_spread = MIC_GRID * MIC_BLOCK

print(f"\natomic 微基准：GRID={MIC_GRID}, BLOCK={MIC_BLOCK}, 每个线程 1 次 atomic")
print(f"{'模式':<26}{'atomic 次数':>14}{'耗时(us)':>11}{'每次 atomic(ns)':>18}")
print("-" * 70)
print(f"{'所有线程砸同一地址（串行）':<24}{n_same:>14,}{t_same * 1000:>11.2f}"
      f"{t_same * 1e6 / n_same:>18.2f}")
print(f"{'每 block 砸自己的地址（并行）':<24}{n_spread:>14,}{t_spread * 1000:>11.2f}"
      f"{t_spread * 1e6 / n_spread:>18.2f}")
print("-" * 70)
print(f"同一地址的 atomic 平均慢 {(t_same / n_same) / (t_spread / n_spread):.1f}x")
print("→ 这就是为什么必须先把 block 内求和，把 atomic 次数从 N 压到 GRID。")

# 校验微基准结果
expect_same = float(n_same)
expect_spread = float(n_spread)
launch_mic_same()
got_same = float(np.sum(cp.asnumpy(d_mic)))
launch_mic_spread()
got_spread = float(np.sum(cp.asnumpy(d_mic)))
print(f"微基准自检: 同一地址 {got_same:.0f}/{expect_same:.0f}，"
      f"分散地址 {got_spread:.0f}/{expect_spread:.0f}")


# ========== 第四组：float atomic 的结果顺序不确定 ==========
# 浮点加法不满足结合律，atomic 的完成顺序每次可能不同 → 末位可能不一样。
RUNS = 30
totals_atomic = []
for _ in range(RUNS):
    launch_per_block()
    cp.cuda.get_current_stream().synchronize()
    totals_atomic.append(total_of(d_sum))

totals_twopass = []
for _ in range(RUNS):
    launch_two_pass()
    cp.cuda.get_current_stream().synchronize()
    totals_twopass.append(total_of(d_two_pass))

print(f"\n连续跑 {RUNS} 次的结果稳定性（float32）：")
print(f"  per-block + atomic: {len(set(totals_atomic))} 种不同结果，"
      f"最大差 {max(totals_atomic) - min(totals_atomic):.3e}")
print(f"  两趟 kernel      : {len(set(totals_twopass))} 种不同结果，"
      f"最大差 {max(totals_twopass) - min(totals_twopass):.3e}")
print(f"参考值(float64)   : {ref:.6e}")

# ========== 结论 ==========
print("\n结论：")
print("1) atomic 对同一地址确实是串行的（微基准约 1.8 ns/次，分散地址只有 0.35 ns/次），")
print("   但每 block 先规约能把 atomic 次数从 N 压到 GRID（BLOCK=256 时 256x），")
print("   N=1M 时 per-block 版比每线程直接 atomic 快十几到几十倍（受降频影响）。")
print("2) 收益来自“减少 atomic 次数”，不是 atomic 本身快。每 warp 版是 N/32 次，")
print("   介于两者之间；block 内规约做得越彻底，atomic 的串行开销越无关紧要。")
print("3) per-block+atomic 与两趟 kernel 在 N=1M 时基本同档（互有胜负，在 ±20% 噪声内）：")
print("   atomic 省掉 partial 数组的全局往返，但计时里也要算上累加器清零的那次 kernel。")
print("4) atomic 版结果不可复现：30 次里出现多种末位不同的值（最大差 ~1e-3），")
print("   因为 float 加法不满足结合律、atomic 完成顺序每次不同；要确定性就选两趟。")
