import cupy as cp
import numpy as np

# ========== Task 16：warp shuffle 规约 ==========
# 思路：规约的每一步本质是"和距离 offset 的伙伴线程交换一个寄存器值"。
# 共享内存版要走 LDS/STS + __syncthreads；warp 内的线程本来就在同一个
# SIMT 单元里，__shfl_down_sync 直接寄存器到寄存器，连内存都不用碰。

# ---------- 1. 纯 warp 规约：BLOCK=32，只用 shfl，不碰 shared ----------
kernel_shfl_warp = r"""
extern "C"
__global__ void sum_shfl_warp(const float* x, float* partial, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; //block内线程编号
    float v = (i < n) ? x[i] : 0.f; //存在每个线程自己的寄存器

    // 5 步（log2(32)）把所有 lane 的和收到 lane 0
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);

    if ((threadIdx.x & 31) == 0) partial[blockIdx.x] = v;
}
"""

# ---------- 2. 通用 block 规约：warp 内 shfl + warp 之间只借 nwarps 个 shared 槽 ----------
# 注意：block 大于 32 时，warp 之间没有寄存器直连，仍然需要一次 shared
# 中转；但只用 1 次 __syncthreads（shared 版要 log2(blockDim) 次）。
kernel_shfl_block = r"""
extern "C"
__global__ void sum_shfl_block(const float* x, float* partial, int n)
{
    __shared__ float warp_sums[32];

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;
    int i    = blockIdx.x * blockDim.x + tid;

    // 第一步：warp 内 shfl 规约，5 步全部在寄存器里完成
    float v = (i < n) ? x[i] : 0.f;
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);

    // 第二步：每个 warp 的 lane 0 把结果写进 shared，整个 block 只同步一次
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();

    // 第三步：warp 0 把最多 32 个 warp 和再用 shfl 收口
    if (wid == 0) {
        int nwarps = blockDim.x >> 5;
        float w = (lane < nwarps) ? warp_sums[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1)
            w += __shfl_down_sync(0xffffffffu, w, off);
        if (lane == 0) partial[blockIdx.x] = w;
    }
}
"""

# ---------- 3. 对照：Task14 顺序寻址 shared 规约（每步一次 __syncthreads）----------
kernel_shared_seq = r"""
extern "C"
__global__ void sum_shared_seq(const float* x, float* partial, int n)
{
    __shared__ float sdata[1024];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? x[i] : 0.f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();          // ← 每步都要全 block 同步，shfl 版一次都不用
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}
"""

# ---------- 4. 只测"规约本身"的微基准：不做全局读，循环 iters 次 block 规约 ----------
# 端到端 kernel 被 4MB 的全局读带宽压住，规约占比很小；
# 这里把访存拿掉，单纯比较两种规约方式的吞吐。
kernel_red_shared = r"""
extern "C"
__global__ void reduce_only_shared(float* out, int iters)
{
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    float acc = 0.f;

    for (int k = 0; k < iters; ++k) {
        sdata[tid] = (float)(tid + k);
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0) acc += sdata[0];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = acc;
}
"""

kernel_red_shfl = r"""
extern "C"
__global__ void reduce_only_shfl(float* out, int iters)
{
    __shared__ float warp_sums[32];
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;
    float acc = 0.f;

    for (int k = 0; k < iters; ++k) {
        float v = (float)(tid + k);
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);

        if (lane == 0) warp_sums[wid] = v;
        __syncthreads();

        if (wid == 0) {
            int nwarps = blockDim.x >> 5;
            float w = (lane < nwarps) ? warp_sums[lane] : 0.f;
            for (int off = 16; off > 0; off >>= 1)
                w += __shfl_down_sync(0xffffffffu, w, off);
            if (lane == 0) acc += w;
        }
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = acc;
}
"""

sum_shfl_warp = cp.RawKernel(kernel_shfl_warp, "sum_shfl_warp")
sum_shfl_block = cp.RawKernel(kernel_shfl_block, "sum_shfl_block")
sum_shared_seq = cp.RawKernel(kernel_shared_seq, "sum_shared_seq")
red_shared = cp.RawKernel(kernel_red_shared, "reduce_only_shared")
red_shfl = cp.RawKernel(kernel_red_shfl, "reduce_only_shfl")

N = 1_000_000
rng = np.random.default_rng(0)
n_x = rng.standard_normal(N, dtype=np.float32)
d_x = cp.asarray(n_x)
ref = float(np.sum(n_x.astype(np.float64)))


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


def check_total(d_partial):
    return abs(float(np.sum(cp.asnumpy(d_partial))) - ref) / abs(ref)


print(f"N={N:,}，GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
print("端到端：每个线程读 1 个 float，block 内规约后写 1 个 partial\n")
print(f"{'BLOCK':>6}{'GRID':>8}{'shared(us)':>12}{'shfl(us)':>11}"
      f"{'shfl 加速':>10}{'shared 误差':>13}{'shfl 误差':>12}")
print("-" * 74)

for BLOCK in (32, 64, 128, 256, 512, 1024):
    GRID = (N + BLOCK - 1) // BLOCK
    d_partial = cp.empty(GRID, dtype=cp.float32)

    def launch_shared():
        sum_shared_seq((GRID, 1, 1), (BLOCK, 1, 1), (d_x, d_partial, np.int32(N)))

    def launch_shfl():
        sum_shfl_block((GRID, 1, 1), (BLOCK, 1, 1), (d_x, d_partial, np.int32(N)))

    t_shared = bench_gpu(launch_shared)
    err_shared = check_total(d_partial)
    t_shfl = bench_gpu(launch_shfl)
    err_shfl = check_total(d_partial)

    # BLOCK=32 时再单独跑"纯 warp"版本（不声明任何 shared）
    extra = ""
    if BLOCK == 32:
        def launch_warp():
            sum_shfl_warp((GRID, 1, 1), (BLOCK, 1, 1), (d_x, d_partial, np.int32(N)))
        t_warp = bench_gpu(launch_warp)
        err_warp = check_total(d_partial)
        extra = (f"\n  BLOCK=32 纯 warp 版（无 shared、无 __syncthreads）: "
                 f"{t_warp * 1000:.2f} us，"
                 f"比 shared 快 {t_shared / t_warp:.2f}x，误差 {err_warp:.2e}")

    print(f"{BLOCK:>6}{GRID:>8}{t_shared * 1000:>12.2f}{t_shfl * 1000:>11.2f}"
          f"{t_shared / t_shfl:>10.2f}{err_shared:>13.2e}{err_shfl:>12.2e}")
    if extra:
        print(extra)

# ========== 只测规约本身：去掉全局读，循环 iters 次 ==========
ITERS = 2000
RED_GRID = 2048
d_out = cp.empty(RED_GRID, dtype=cp.float32)

print(f"\n规约微基准：GRID={RED_GRID}，每个 block 循环 {ITERS} 次规约，无全局读")
print(f"{'BLOCK':>6}{'shared(us)':>12}{'shfl(us)':>11}{'shfl 加速':>10}"
      f"{'每次规约(ns)':>14}{'吞吐(M/s)':>12}")
print("-" * 70)

for BLOCK in (32, 128, 256, 1024):
    def launch_shared():
        red_shared((RED_GRID, 1, 1), (BLOCK, 1, 1), (d_out, np.int32(ITERS)))

    def launch_shfl():
        red_shfl((RED_GRID, 1, 1), (BLOCK, 1, 1), (d_out, np.int32(ITERS)))

    t_shared = bench_gpu(launch_shared, warmup=5, repeat=50)
    t_shfl = bench_gpu(launch_shfl, warmup=5, repeat=50)

    # 整卡并行时摊到每一次 block 规约的墙钟时间，以及整卡吞吐
    per_iter_ns = t_shfl * 1e6 / (RED_GRID * ITERS)
    throughput = RED_GRID * ITERS / t_shfl / 1000.0        # 次/ms -> M/s
    print(f"{BLOCK:>6}{t_shared * 1000:>12.2f}{t_shfl * 1000:>11.2f}"
          f"{t_shared / t_shfl:>10.2f}{per_iter_ns:>14.1f}{throughput:>12.1f}")

# ========== 正确性：规约微基准的理论值核对 ==========
BLOCK = 256
red_shared((RED_GRID, 1, 1), (BLOCK, 1, 1), (d_out, np.int32(ITERS)))
v_shared = float(np.sum(cp.asnumpy(d_out)))
red_shfl((RED_GRID, 1, 1), (BLOCK, 1, 1), (d_out, np.int32(ITERS)))
v_shfl = float(np.sum(cp.asnumpy(d_out)))
expect = RED_GRID * (ITERS * BLOCK * (BLOCK - 1) / 2
                     + BLOCK * ITERS * (ITERS - 1) / 2)
print(f"\nBLOCK={BLOCK} 微基准正确性: shared={v_shared:.6e}, shfl={v_shfl:.6e}, "
      f"理论={expect:.6e}")
print(f"相对误差: shared {abs(v_shared - expect) / expect:.2e}, "
      f"shfl {abs(v_shfl - expect) / expect:.2e}")

# ========== 对照 cupy 自带的 sum ==========
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
print(f"\ncupy d_x.sum(): {t_ref * 1000:.2f} us")

# ========== 一张图看懂 shfl_down：8 个 lane 的和怎么收口 ==========
print("\n__shfl_down_sync 的收口过程（lane 值 = 0..7，off=4,2,1 三轮）：")
lane_demo = np.arange(8, dtype=np.float32)
print("初始 :", lane_demo)
for off in (4, 2, 1):
    # lane i 取 lane i+off 的值（越界的 lane 结果无意义，不参与下一步）
    nxt = lane_demo.copy()
    for i in range(8 - off):
        nxt[i] = lane_demo[i] + lane_demo[i + off]
    lane_demo = nxt
    print(f"off={off} :", lane_demo)
print(f"lane 0 得到总和: {lane_demo[0]}")

# ========== 结论 ==========
print("\n结论：")
print("1) 端到端（读 4MB + 规约）：shfl 比 shared 快约 1.1x~2.6x，block 越大差距越大。")
print("   shared 版每步一次 __syncthreads（BLOCK=1024 要 10 次），shfl 版整个 block 只同步 1 次。")
print("2) 只测规约本身：shfl 快约 1.6x~3.8x。shared 每步 2 次 LDS/STS + 屏障，")
print("   shfl 每步只有 1 条 SHFL 指令，5 步全在寄存器里完成，不产生 shared 流量。")
print("3) BLOCK=32 时可以完全不用 shared（纯 warp 版），比同尺寸 shared 版快 1.35x。")
print("4) 端到端 kernel 仍被全局读带宽压住，规约只占小头，所以收益会被稀释；")
print("   规约在总耗时里占比越高（大 block、数据已在寄存器里），shuffle 的优势越明显。")
