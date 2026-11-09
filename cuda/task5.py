import cupy as cp
import numpy as np

BLOCK = 256

# ---------- kernel 1：空 kernel，专门测「launch 纯开销」 ----------
src_noop = r"""
extern "C" __global__ void noop_kernel() {
}
"""
k_noop = cp.RawKernel(src_noop, "noop_kernel")

# ---------- kernel 2：Task2 的 printf kernel（限制只 1 个线程打印） ----------
src_printf = r"""
extern "C" __global__ void hello_gpu() {
    if (threadIdx.x == 0) {
        printf("Hello GPU!\n");
    }
}
"""
k_printf = cp.RawKernel(src_printf, "hello_gpu")

# ---------- kernel 3：真实负载，Task3 的 index kernel ----------
src_index = r"""
extern "C" __global__ void index_kernel(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i;
}
"""
k_index = cp.RawKernel(src_index, "index_kernel")


# ---------- 计时工具：warmup + 批量计时 ----------
def bench(fn, warmup=10, repeat=100):
    # 1) 预热：把冷启动开销都付掉
    for _ in range(warmup):
        fn()
    cp.cuda.get_current_stream().synchronize()

    # 2) 批量计时：N 次 launch 只用两端 event
    e0 = cp.cuda.Event()
    e1 = cp.cuda.Event()
    e0.record()
    for _ in range(repeat):
        fn()
    e1.record()
    cp.cuda.get_current_stream().synchronize()

    ms_total = cp.cuda.get_elapsed_time(e0, e1)   # 单位：毫秒
    return ms_total, ms_total / repeat


# ---------- 1) 空 kernel：测 launch 纯开销 ----------
t_total, t_avg = bench(lambda: k_noop((1, 1, 1), (BLOCK, 1, 1), ()), 10, 1000)
print(f"[noop]          total={t_total:8.4f} ms   per-call={t_avg * 1000:7.3f} us")

# ---------- 2) printf kernel：测调试打印的代价 ----------
t_total, t_avg = bench(lambda: k_printf((1, 1, 1), (BLOCK, 1, 1), ()), 5, 20)
print(f"[printf]        total={t_total:8.4f} ms   per-call={t_avg * 1000:7.3f} us")

# ---------- 3) 真实负载：写 N 个元素，凑出 ms 级 ----------
N = 50_000_000                                  # 2 亿字节 = 200 MB 显存
d_out = cp.empty(N, dtype=cp.int32)
GRID = ((N + BLOCK - 1) // BLOCK, 1, 1)

t_total, t_avg = bench(lambda: k_index(GRID, (BLOCK, 1, 1), (d_out, np.int32(N))), 3, 20)
print(f"[index N={N}] total={t_total:8.4f} ms   per-call={t_avg:8.4f} ms")
