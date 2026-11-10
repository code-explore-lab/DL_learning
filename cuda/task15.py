import cupy as cp
import numpy as np

# ========== Task 14 基础版：相邻两元素相加 + 取模选活跃线程 ==========
kernel_basic = r"""
extern "C"
__global__ void sum_shared_basic(const float* x, float* partial, int n)
{
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? x[i] : 0.f;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s <<= 1) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}
"""

# ========== Task 14 对照：顺序寻址（tid < s，tid + s）==========
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
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}
"""

# ========== Task 15 树状规约：相邻子节点相加 sdata[2*tid] + sdata[2*tid+1] ==========
# 关键：不能原地写回 sdata[tid]。同一步里 warp A 写 sdata[tid] 时，
# warp B 可能正在读同一个槽（读集合 2*tid、2*tid+1 会覆盖写集合 tid），
# 跨 warp 没有顺序保证，会读到被覆盖的值。用双缓冲 src/dst 规避。
# 附带收益：相邻的两个 float 会被编译器合成一条 LDS.64（已用 nvdisasm 验证），
# 所以即使有 2-way bank conflict，也比 Task14 的两次独立 LDS 快。
kernel_tree = r"""
extern "C"
__global__ void sum_shared_tree(const float* x, float* partial, int n)
{
    __shared__ float buf0[256];
    __shared__ float buf1[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    buf0[tid] = (i < n) ? x[i] : 0.f;
    __syncthreads();

    float* src = buf0;
    float* dst = buf1;
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            dst[tid] = src[2 * tid] + src[2 * tid + 1];   // 两个相邻子节点
        }
        __syncthreads();                                  // src 只读、dst 只写，一步一同步
        float* t = src; src = dst; dst = t;
    }
    if (tid == 0) partial[blockIdx.x] = src[0];
}
"""

# ========== 错误示范：直接原地写回，存在跨 warp 读写竞争 ==========
kernel_tree_racy = r"""
extern "C"
__global__ void sum_shared_tree_racy(const float* x, float* partial, int n)
{
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? x[i] : 0.f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = sdata[2 * tid] + sdata[2 * tid + 1];   // ← 竞争
        }
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}
"""

sum_basic = cp.RawKernel(kernel_basic, "sum_shared_basic")
sum_seq = cp.RawKernel(kernel_seq, "sum_shared_seq")
sum_tree = cp.RawKernel(kernel_tree, "sum_shared_tree")
sum_tree_racy = cp.RawKernel(kernel_tree_racy, "sum_shared_tree_racy")

BLOCK = 256
N = 1_000_000
GRID = (N + BLOCK - 1) // BLOCK

rng = np.random.default_rng(0)
n_x = rng.standard_normal(N, dtype=np.float32)
d_x = cp.asarray(n_x)
ref = float(np.sum(n_x.astype(np.float64)))

d_partial = cp.empty(GRID, dtype=cp.float32)


def bench_gpu(launch, warmup=10, repeat=300):
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


def result_and_error(launch):
    launch()
    total = float(np.sum(cp.asnumpy(d_partial)))   # CPU 再 reduce
    return total, abs(total - ref) / abs(ref)


def make_launcher(kernel):
    return lambda: kernel((GRID, 1, 1), (BLOCK, 1, 1),
                          (d_x, d_partial, np.int32(N)))


launchers = {
    "Task14 基础版 (取模分叉)": make_launcher(sum_basic),
    "Task14 顺序寻址 (tid+s)": make_launcher(sum_seq),
    "Task15 树状 (2tid,2tid+1)": make_launcher(sum_tree),
    "错误示范:原地写回": make_launcher(sum_tree_racy),
}

print(f"N={N:,}, BLOCK={BLOCK}, GRID={GRID}, 每 block {BLOCK} 个元素\n")
print(f"{'方式':<28}{'kernel(ms)':>12}{'vs 基础版':>12}{'相对误差':>12}")
print("-" * 66)

times = {}
for name, launch in launchers.items():
    _, err = result_and_error(launch)
    t = bench_gpu(launch)
    times[name] = t
    base = times["Task14 基础版 (取模分叉)"]
    print(f"{name:<28}{t:>12.4f}{t / base:>12.2f}{err:>12.2e}")

print("-" * 66)
print("原地写回那行误差极大：同一步内 warp 之间的读写重叠，结果不可信。")


def bench_cupy_sum(warmup=10, repeat=300):
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
print(f"树状规约 / cupy: {times['Task15 树状 (2tid,2tid+1)'] / t_ref:.2f}x")
