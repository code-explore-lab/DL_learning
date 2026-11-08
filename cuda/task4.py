import cupy as cp
import numpy as np

# -------------------------- 配置参数 --------------------------
N = 1_000_000
BLOCK = 256
grid = ((N + BLOCK - 1) // BLOCK, 1, 1)

# -------------------------- CUDA C Kernel：每个元素 +1 --------------------------
kernel_code = r"""
extern "C" __global__ void add1_kernel(int* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] + 1;
    }
}
"""
add1_kernel = cp.RawKernel(kernel_code, "add1_kernel")

# -------------------------- 准备CPU侧数据（Host内存，numpy） --------------------------
h_in_np = np.arange(N, dtype=np.int32)   # CPU内存
h_ref = h_in_np + 1                     # 预期正确结果，用来对比

# -------------------------- 分配显存、准备GPU数组 --------------------------
d_in = cp.empty(N, dtype=cp.int32)
d_out = cp.empty(N, dtype=cp.int32)

# CUDA Event：GPU高精度计时器（GPU硬件时钟，精度远高于Python time）
start_h2d = cp.cuda.Event()
start_kernel = cp.cuda.Event()
start_d2h = cp.cuda.Event()
end_all = cp.cuda.Event()

stream = cp.cuda.get_current_stream()

# ========== 开始计时流程 ==========
start_h2d.record(stream)
# Host -> Device 拷贝 （等价 cudaMemcpyHostToDevice）
d_in = cp.array(h_in_np)
start_kernel.record(stream)

# Launch kernel，GPU计算
add1_kernel(grid, (BLOCK, 1, 1), (d_in, d_out, np.int32(N)))
start_d2h.record(stream)

# Device -> Host 拷贝，拿回CPU
h_result = d_out.get()
end_all.record(stream)

# 等待所有任务完成，才能读取时间
stream.synchronize()
## [盖章A → H2D拷贝 → 盖章B → kernel计算 → 盖章C → D2H拷贝 → 盖章D]
# 读取时间，单位：毫秒 ms
t_h2d = cp.cuda.get_elapsed_time(start_h2d, start_kernel)
t_kernel = cp.cuda.get_elapsed_time(start_kernel, start_d2h)
t_d2h = cp.cuda.get_elapsed_time(start_d2h, end_all)
t_total = cp.cuda.get_elapsed_time(start_h2d, end_all)

# -------------------------- 校验结果 --------------------------
ok = np.array_equal(h_result, h_ref)

# -------------------------- 打印报告 --------------------------
print(f"N = {N}")
print(f"✅ 结果正确？ {ok}")
print("===== 时间统计（ms） =====")
print(f"Host->Device拷贝 : {t_h2d:.4f} ms")
print(f"Kernel计算(+1)   : {t_kernel:.4f} ms")
print(f"Device->Host拷贝 : {t_d2h:.4f} ms")
print(f"总耗时           : {t_total:.4f} ms")
print(f"拷贝占总时间比例 : {(t_h2d + t_d2h)/t_total *100:.2f} %")
