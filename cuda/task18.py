import cupy as cp
import numpy as np

# ========== Task 18：Max / Min 规约 + argmax ==========
# 从 sum 规约改成 max/min，只动两处：
#   1) 单位元：sum 是 0，max 是 -FLT_MAX，min 是 +FLT_MAX；
#   2) 结合运算：v += other 变成 v = fmaxf(v, other) / fminf(v, other)。
# 规约树、warp shuffle、跨 warp 用 shared 收口的骨架完全不变。
#
# argmax 额外带一个下标：每步 shuffle 要同时交换 (value, index)。
# 比较规则用字典序：value 大的赢；value 相等时 index 小的赢
# （和 numpy.argmax 返回第一个最大值的行为一致）。

# ---------- 1. 对照：shared 内存树状 max（Task14 的顺序寻址版）----------
kernel_max_shared = r"""
extern "C"
__global__ void max_shared(const float* x, float* partial, int n)
{
    __shared__ float sdata[1024];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? x[i] : -3.402823466e38f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);   // ← sum 时是 +=
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}
"""

# ---------- 2. warp shuffle max：block 内规约，写 partial ----------
kernel_max_shfl = r"""
extern "C"
__global__ void max_shfl_block(const float* x, float* partial, int n)
{
    __shared__ float warp_vals[32];

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;
    int i    = blockIdx.x * blockDim.x + tid;

    float v = (i < n) ? x[i] : -3.402823466e38f;      // max 的单位元
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));

    if (lane == 0) warp_vals[wid] = v;
    __syncthreads();

    if (wid == 0) {
        int nwarps = blockDim.x >> 5;
        float w = (lane < nwarps) ? warp_vals[lane] : -3.402823466e38f;
        for (int off = 16; off > 0; off >>= 1)
            w = fmaxf(w, __shfl_down_sync(0xffffffffu, w, off));
        if (lane == 0) partial[blockIdx.x] = w;
    }
}
"""

# ---------- 3. warp shuffle min：只把 fmaxf/单位元换掉 ----------
kernel_min_shfl = r"""
extern "C"
__global__ void min_shfl_block(const float* x, float* partial, int n)
{
    __shared__ float warp_vals[32];

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;
    int i    = blockIdx.x * blockDim.x + tid;

    float v = (i < n) ? x[i] : 3.402823466e38f;       // min 的单位元
    for (int off = 16; off > 0; off >>= 1)
        v = fminf(v, __shfl_down_sync(0xffffffffu, v, off));

    if (lane == 0) warp_vals[wid] = v;
    __syncthreads();

    if (wid == 0) {
        int nwarps = blockDim.x >> 5;
        float w = (lane < nwarps) ? warp_vals[lane] : 3.402823466e38f;
        for (int off = 16; off > 0; off >>= 1)
            w = fminf(w, __shfl_down_sync(0xffffffffu, w, off));
        if (lane == 0) partial[blockIdx.x] = w;
    }
}
"""

# ---------- 4. argmax：每步同时 shuffle value 和 index ----------
kernel_argmax_shfl = r"""
extern "C"
__global__ void argmax_shfl_block(const float* x, float* out_val,
                                  int* out_idx, int n)
{
    __shared__ float warp_vals[32];
    __shared__ int   warp_idxs[32];

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;
    int i    = blockIdx.x * blockDim.x + tid;

    // 越界线程用单位元 + 最大下标，保证不会在并列时“赢”
    float v = (i < n) ? x[i] : -3.402823466e38f;
    int   idx = (i < n) ? i : 0x7fffffff;

    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, v, off);   // 换值
        int   oi = __shfl_down_sync(0xffffffffu, idx, off); // 换下标
        if (ov > v || (ov == v && oi < idx)) {              // 大者胜；并列取小下标
            v = ov;
            idx = oi;
        }
    }

    if (lane == 0) {
        warp_vals[wid] = v;
        warp_idxs[wid] = idx;
    }
    __syncthreads();

    if (wid == 0) {
        int nwarps = blockDim.x >> 5;
        float w  = (lane < nwarps) ? warp_vals[lane] : -3.402823466e38f;
        int   wi = (lane < nwarps) ? warp_idxs[lane] : 0x7fffffff;
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, w, off);
            int   oi = __shfl_down_sync(0xffffffffu, wi, off);
            if (ov > w || (ov == w && oi < wi)) {
                w  = ov;
                wi = oi;
            }
        }
        if (lane == 0) {
            out_val[blockIdx.x] = w;
            out_idx[blockIdx.x] = wi;      // 全局下标
        }
    }
}
"""

max_shared = cp.RawKernel(kernel_max_shared, "max_shared")
max_shfl = cp.RawKernel(kernel_max_shfl, "max_shfl_block")
min_shfl = cp.RawKernel(kernel_min_shfl, "min_shfl_block")
argmax_shfl = cp.RawKernel(kernel_argmax_shfl, "argmax_shfl_block")

N = 1_000_000
BLOCK = 256
GRID = (N + BLOCK - 1) // BLOCK

rng = np.random.default_rng(0)
n_x = rng.standard_normal(N, dtype=np.float32)
d_x = cp.asarray(n_x)

ref_max = float(np.max(n_x))
ref_min = float(np.min(n_x))
ref_argmax = int(np.argmax(n_x))            # numpy 也是并列取第一个

d_partial = cp.empty(GRID, dtype=cp.float32)
d_vals = cp.empty(GRID, dtype=cp.float32)
d_idxs = cp.empty(GRID, dtype=cp.int32)


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


ARG = np.int32(N)
GRID_T = (GRID, 1, 1)
BLOCK_T = (BLOCK, 1, 1)

def launch_max_shared():
    max_shared(GRID_T, BLOCK_T, (d_x, d_partial, ARG))

def launch_max_shfl():
    max_shfl(GRID_T, BLOCK_T, (d_x, d_partial, ARG))

def launch_min_shfl():
    min_shfl(GRID_T, BLOCK_T, (d_x, d_partial, ARG))

def launch_argmax():
    argmax_shfl(GRID_T, BLOCK_T, (d_x, d_vals, d_idxs, ARG))


print(f"N={N:,}, BLOCK={BLOCK}, GRID={GRID}，"
      f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}\n")

# ========== 正确性 + 计时 ==========
launch_max_shared()
got_max_shared = float(np.max(cp.asnumpy(d_partial)))

launch_max_shfl()
got_max_shfl = float(np.max(cp.asnumpy(d_partial)))

launch_min_shfl()
got_min = float(np.min(cp.asnumpy(d_partial)))

launch_argmax()
vals = cp.asnumpy(d_vals)
idxs = cp.asnumpy(d_idxs)
b = int(np.argmax(vals))                    # 先找最大 partial 所在的 block
got_argmax = int(idxs[b])
got_argmax_val = float(n_x[got_argmax])     # 用原数组核对取回的值

t_max_shared = bench_gpu(launch_max_shared)
t_max_shfl = bench_gpu(launch_max_shfl)
t_min_shfl = bench_gpu(launch_min_shfl)
t_argmax = bench_gpu(launch_argmax)

print(f"{'kernel':<30}{'结果':>14}{'耗时(us)':>12}")
print("-" * 58)
print(f"{'shared max（对照）':<26}{got_max_shared:>14.6f}{t_max_shared * 1000:>12.2f}")
print(f"{'shfl max':<26}{got_max_shfl:>14.6f}{t_max_shfl * 1000:>12.2f}")
print(f"{'shfl min':<26}{got_min:>14.6f}{t_min_shfl * 1000:>12.2f}")
print(f"{'shfl argmax（值）':<26}{got_argmax_val:>14.6f}{t_argmax * 1000:>12.2f}")
print(f"{'shfl argmax（下标）':<26}{got_argmax:>14d}")
print("-" * 58)
print(f"shfl max 比 shared max 快 {t_max_shared / t_max_shfl:.2f}x")
print(f"argmax 比 max 多花 {t_argmax / t_max_shfl:.2f}x"
      f"（多一次 index shuffle + 比较）")

print(f"\n正确性：")
print(f"  max   : {got_max_shfl:.9g}  参考 {ref_max:.9g}  "
      f"{'OK' if got_max_shfl == ref_max else 'FAIL'}")
print(f"  min   : {got_min:.9g}  参考 {ref_min:.9g}  "
      f"{'OK' if got_min == ref_min else 'FAIL'}")
print(f"  argmax: {got_argmax}  参考 {ref_argmax}  "
      f"{'OK' if got_argmax == ref_argmax else 'FAIL'}"
      f"（值 {got_argmax_val:.9g}）")


# ========== 并列时取最小下标：和 np.argmax 对齐 ==========
# 在一个 block 内制造跨 warp 的并列：下标 20 和 150 都是 9.0。
DEMO_N = 200
n_demo = np.zeros(DEMO_N, dtype=np.float32)
n_demo[20] = 9.0
n_demo[150] = 9.0
d_demo = cp.asarray(n_demo)
d_demo_vals = cp.empty(1, dtype=cp.float32)
d_demo_idxs = cp.empty(1, dtype=cp.int32)

argmax_shfl((1, 1, 1), (BLOCK, 1, 1),
            (d_demo, d_demo_vals, d_demo_idxs, np.int32(DEMO_N)))
demo_idx = int(cp.asnumpy(d_demo_idxs)[0])
demo_val = float(cp.asnumpy(d_demo_vals)[0])
print(f"\n并列测试：下标 20 和 150 的值都是 9.0 -> 返回下标 {demo_idx}"
      f"（np.argmax={int(np.argmax(n_demo))}）")
print(f"  跨 warp 并列也能取到最小下标: {'OK' if demo_idx == 20 else 'FAIL'}"
      f"，值 {demo_val:g}")


# ========== 对照 cupy ==========
def bench_cupy(fn, warmup=10, repeat=200):
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


t_cp_max = bench_cupy(lambda: d_x.max())
t_cp_min = bench_cupy(lambda: d_x.min())
t_cp_argmax = bench_cupy(lambda: d_x.argmax())
print(f"\ncupy: max {t_cp_max * 1000:.2f} us，min {t_cp_min * 1000:.2f} us，"
      f"argmax {t_cp_argmax * 1000:.2f} us")
print(f"我们的 max 耗时 / cupy = {t_max_shfl / t_cp_max:.2f}x，"
      f"argmax / cupy = {t_argmax / t_cp_argmax:.2f}x（<1 表示我们更快）")

# ========== 结论 ==========
print("\n结论：")
print("1) sum → max/min 只改单位元和结合运算（+ 换成 fmaxf/fminf），")
print("   shuffle 规约的骨架、跨 warp 收口方式完全复用。")
print("2) argmax 的代价是每步要多 shuffle 一个 index、多做一次比较；")
print("   并列时必须定义规则（这里取最小下标），否则跨 warp 合并时结果不稳定。")
print("3) max/min 满足结合律和交换律，规约顺序不影响结果，所以结果确定、可复现；")
print("   这也是它能用 atomicMax 做跨 block 收口的原因")
print("   （atomicMax 原生只支持整型，浮点要么用 CAS 循环，要么把非负 float 当 int 比较）。")
