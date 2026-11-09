import cupy as cp
import numpy as np

# ========== 正确的二维索引：x 方向对应列，y 方向对应行 ==========
kernel_code = r"""
extern "C"
__global__ void matadd(const float* A, const float* B, float* C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;   // 行主序展平
        C[idx] = A[idx] + B[idx];
    }
}
"""

# ========== 反例：把 row/col 的映射交换，看合并访问被打断的代价 ==========
kernel_swapped_code = r"""
extern "C"
__global__ void matadd_swapped(const float* A, const float* B, float* C, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}
"""

matadd = cp.RawKernel(kernel_code, "matadd")
matadd_swapped = cp.RawKernel(kernel_swapped_code, "matadd_swapped")

ROWS, COLS = 1024, 1024
BX, BY = 16, 16                      # 每个 block 16x16 = 256 线程
GRID = ((COLS + BX - 1) // BX, (ROWS + BY - 1) // BY, 1)
BLOCK = (BX, BY, 1)

d_A = cp.random.randn(ROWS, COLS, dtype=cp.float32)
d_B = cp.random.randn(ROWS, COLS, dtype=cp.float32)
d_C = cp.empty_like(d_A)
d_C_bad = cp.empty_like(d_A)


def bench(fn, warmup=10, repeat=200):
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


print(f"矩阵 {ROWS}x{COLS}, block={BX}x{BY}, grid={GRID[0]}x{GRID[1]}")

t_ok = bench(lambda: matadd(GRID, BLOCK, (d_A, d_B, d_C, np.int32(ROWS), np.int32(COLS))))
t_bad = bench(lambda: matadd_swapped(GRID, BLOCK, (d_A, d_B, d_C_bad, np.int32(ROWS), np.int32(COLS))))
t_cupy = bench(lambda: cp.add(d_A, d_B, out=d_C))

print("\n=== 耗时 ===")
print(f"正确二维索引 : {t_ok * 1000:7.2f} us")
print(f"交换 row/col : {t_bad * 1000:7.2f} us")
print(f"cupy add     : {t_cupy * 1000:7.2f} us")
print(f"反例/正例    : {t_bad / t_ok:.2f}x")

# ========== 验收：和 numpy A + B 一致 ==========
A_np = cp.asnumpy(d_A)
B_np = cp.asnumpy(d_B)
C_np = cp.asnumpy(d_C)

np.testing.assert_allclose(C_np, A_np + B_np, rtol=1e-5, atol=1e-6)
print("\n与 numpy A + B 一致（正确索引版）")

# 反例在数值上也是对的，只是慢
C_bad_np = cp.asnumpy(d_C_bad)
np.testing.assert_allclose(C_bad_np, A_np + B_np, rtol=1e-5, atol=1e-6)
print("交换版数值同样正确，慢的是访存模式")

# ========== 补充：数据超过 L2 后，合并访问的差距才会显出来 ==========
R2 = C2 = 4096                       # 每个矩阵 64 MB，三份远大于 L2
GRID2 = ((C2 + BX - 1) // BX, (R2 + BY - 1) // BY, 1)
a2 = cp.random.randn(R2, C2, dtype=cp.float32)
b2 = cp.random.randn(R2, C2, dtype=cp.float32)
c2_ok = cp.empty_like(a2)
c2_bad = cp.empty_like(a2)

t2_ok = bench(lambda: matadd(GRID2, BLOCK, (a2, b2, c2_ok, np.int32(R2), np.int32(C2))),
              warmup=5, repeat=30)
t2_bad = bench(lambda: matadd_swapped(GRID2, BLOCK, (a2, b2, c2_bad, np.int32(R2), np.int32(C2))),
               warmup=5, repeat=30)
t2_cupy = bench(lambda: cp.add(a2, b2, out=c2_ok), warmup=5, repeat=30)

print(f"\n=== {R2}x{C2}（超过 L2）===")
print(f"正确二维索引 : {t2_ok:.4f} ms")
print(f"交换 row/col : {t2_bad:.4f} ms")
print(f"cupy add     : {t2_cupy:.4f} ms")
print(f"反例/正例    : {t2_bad / t2_ok:.2f}x")
