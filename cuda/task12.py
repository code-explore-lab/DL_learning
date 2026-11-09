import time

import cupy as cp
import numpy as np
import torch

# ==================== kernel 定义（Task 6-11） ====================
k_add = cp.RawKernel(r"""
extern "C" __global__ void k_add(const float* a, const float* b, float* out, int n)
{
    int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += stride) out[i] = a[i] + b[i];
}
""", "k_add")

k_axpy = cp.RawKernel(r"""
extern "C" __global__ void k_axpy(float a, const float* x, float* y, int n)
{
    int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += stride) y[i] = a * x[i] + y[i];
}
""", "k_axpy")

k_relu = cp.RawKernel(r"""
extern "C" __global__ void k_relu(const float* x, float* out, int n)
{
    int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += stride) out[i] = fmaxf(0.f, x[i]);
}
""", "k_relu")

k_sigmoid = cp.RawKernel(r"""
extern "C" __global__ void k_sigmoid(const float* x, float* out, int n)
{
    int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += stride) out[i] = 1.f / (1.f + expf(-x[i]));
}
""", "k_sigmoid")

k_scale = cp.RawKernel(r"""
extern "C" __global__ void k_scale(float a, const float* x, float* out, int n)
{
    int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += stride) out[i] = a * x[i];
}
""", "k_scale")

k_matadd = cp.RawKernel(r"""
extern "C" __global__ void k_matadd(const float* A, const float* B, float* C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}
""", "k_matadd")

# 向量化对照：每个线程一次搬 128 bit（4 个 float），和 torch 的逐元素 kernel 做法一致
k_scale_vec4 = cp.RawKernel(r"""
extern "C" __global__ void k_scale_vec4(float a, const float4* x, float4* out, int n4)
{
    int stride = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n4; i += stride) {
        float4 v = x[i];
        v.x *= a; v.y *= a; v.z *= a; v.w *= a;
        out[i] = v;
    }
}
""", "k_scale_vec4")

BLOCK = 256
N = 20_000_000
GRID = ((N + BLOCK - 1) // BLOCK, 1, 1)
A_VAL = 2.5
ROWS = COLS = 1024
BXY = (16, 16, 1)
GRID2 = ((COLS + 15) // 16, (ROWS + 15) // 16, 1)

# ==================== 数据：numpy 生成一份，三家共用，便于最后对拍 ====================
rng = np.random.default_rng(0)
n_a = rng.standard_normal(N, dtype=np.float32)
n_b = rng.standard_normal(N, dtype=np.float32)
n_x = rng.standard_normal(N, dtype=np.float32)
n_y = rng.standard_normal(N, dtype=np.float32)
n_out = np.empty(N, dtype=np.float32)
n_tmp = np.empty(N, dtype=np.float32)
n_A = rng.standard_normal((ROWS, COLS), dtype=np.float32)
n_B = rng.standard_normal((ROWS, COLS), dtype=np.float32)
n_C = np.empty_like(n_A)

# CuPy / 手写 kernel
d_a = cp.asarray(n_a)
d_b = cp.asarray(n_b)
d_x = cp.asarray(n_x)
d_y = cp.asarray(n_y)
d_out = cp.empty(N, dtype=cp.float32)
d_A = cp.asarray(n_A)
d_B = cp.asarray(n_B)
d_C = cp.empty_like(d_A)

# PyTorch（GPU）
t_a = torch.as_tensor(n_a, device="cuda")
t_b = torch.as_tensor(n_b, device="cuda")
t_x = torch.as_tensor(n_x, device="cuda")
t_y = torch.as_tensor(n_y, device="cuda")
t_out = torch.empty(N, device="cuda", dtype=torch.float32)
t_A = torch.as_tensor(n_A, device="cuda")
t_B = torch.as_tensor(n_B, device="cuda")
t_C = torch.empty_like(t_A)


# ==================== 三种计时器 ====================
def bench_cupy(fn, warmup=5, repeat=50):
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


def bench_torch(fn, warmup=5, repeat=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(repeat):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / repeat


def bench_numpy(fn, warmup=2, repeat=10):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) / repeat * 1000.0


# ==================== 各算子的三种实现 ====================
def np_sigmoid():
    np.negative(n_x, out=n_tmp)
    np.exp(n_tmp, out=n_tmp)
    np.add(n_tmp, 1.0, out=n_tmp)
    np.reciprocal(n_tmp, out=n_out)


ops_1d = [
    ("add      (20M)",
     lambda: k_add(GRID, (BLOCK, 1, 1), (d_a, d_b, d_out, np.int32(N))),
     lambda: np.add(n_a, n_b, out=n_out),
     lambda: torch.add(t_a, t_b, out=t_out)),
    ("axpy     (20M)",
     lambda: k_axpy(GRID, (BLOCK, 1, 1), (cp.float32(A_VAL), d_x, d_y, np.int32(N))),
     lambda: (np.multiply(n_x, A_VAL, out=n_tmp), np.add(n_tmp, n_y, out=n_out)),
     lambda: torch.add(t_y, t_x, alpha=A_VAL, out=t_out)),
    ("relu     (20M)",
     lambda: k_relu(GRID, (BLOCK, 1, 1), (d_x, d_out, np.int32(N))),
     lambda: np.maximum(n_x, 0.0, out=n_out),
     lambda: torch.clamp_min(t_x, 0.0, out=t_out)),
    ("sigmoid  (20M)",
     lambda: k_sigmoid(GRID, (BLOCK, 1, 1), (d_x, d_out, np.int32(N))),
     np_sigmoid,
     lambda: torch.sigmoid(t_x, out=t_out)),
    ("scale    (20M)",
     lambda: k_scale(GRID, (BLOCK, 1, 1), (cp.float32(A_VAL), d_x, d_out, np.int32(N))),
     lambda: np.multiply(n_x, A_VAL, out=n_out),
     lambda: torch.mul(t_x, A_VAL, out=t_out)),
]

ops_2d = [
    ("matadd (1024^2)",
     lambda: k_matadd(GRID2, BXY, (d_A, d_B, d_C, np.int32(ROWS), np.int32(COLS))),
     lambda: np.add(n_A, n_B, out=n_C),
     lambda: torch.add(t_A, t_B, out=t_C)),
]

rows = []
for name, my_fn, np_fn, t_fn in ops_1d:
    rows.append((name,
                 bench_cupy(my_fn), bench_numpy(np_fn), bench_torch(t_fn)))
for name, my_fn, np_fn, t_fn in ops_2d:
    rows.append((name,
                 bench_cupy(my_fn, 5, 200), bench_numpy(np_fn, 2, 50),
                 bench_torch(t_fn, 5, 200)))

# ==================== 输出表格 ====================
print(f"GPU: {torch.cuda.get_device_name(0)}  |  1D N={N:,}  |  2D {ROWS}x{COLS}")
print()
print(f"{'算子':<16}{'手写(ms)':>11}{'numpy(ms)':>12}{'torch(ms)':>11}"
      f"{'numpy÷手写':>12}{'torch÷手写':>12}")
print("-" * 74)
for name, t_my, t_np, t_tt in rows:
    print(f"{name:<16}{t_my:>11.4f}{t_np:>12.4f}{t_tt:>11.4f}"
          f"{t_np / t_my:>11.1f}x{t_tt / t_my:>11.2f}x")
print("-" * 74)
print("numpy÷手写 >1: numpy(CPU) 比手写慢的倍数；torch÷手写 >1: torch 比手写快")

# ==================== 追因：标量 vs float4 向量化（以 scale 为例） ====================
N4 = N // 4
GRID4 = ((N4 + BLOCK - 1) // BLOCK, 1, 1)
t_vec4 = bench_cupy(lambda: k_scale_vec4(GRID4, (BLOCK, 1, 1),
                                         (cp.float32(A_VAL), d_x, d_out, np.int32(N4))))
t_scalar = bench_cupy(lambda: k_scale(GRID, (BLOCK, 1, 1),
                                      (cp.float32(A_VAL), d_x, d_out, np.int32(N))))
t_torch = bench_torch(lambda: torch.mul(t_x, A_VAL, out=t_out))
print(f"\nscale 追因: 标量={t_scalar:.4f} ms  float4={t_vec4:.4f} ms  "
      f"torch={t_torch:.4f} ms")
print(f"标量→float4 提升 {t_scalar / t_vec4:.2f}x")
k_scale_vec4(GRID4, (BLOCK, 1, 1), (cp.float32(A_VAL), d_x, d_out, np.int32(N4)))
cp.testing.assert_allclose(d_out, A_VAL * d_x, rtol=1e-5, atol=1e-6)
print("float4 版本结果正确")

# ==================== 顺手做个正确性对拍：手写 vs torch ====================
k_add(GRID, (BLOCK, 1, 1), (d_a, d_b, d_out, np.int32(N)))
torch.add(t_a, t_b, out=t_out)
cp.testing.assert_allclose(d_out, cp.asarray(t_out), rtol=1e-5, atol=1e-6)
print("add 手写结果与 torch 一致")
