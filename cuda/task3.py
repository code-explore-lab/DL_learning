import cupy as cp 
import numpy as np 

N = 1_000_000
BLOCK = 256 

grid = ((N+BLOCK-1)//BLOCK, 1, 1)

kernel_code = r"""
extern "C" __global__ void index_kernel(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = i;
    }
}
"""
out = cp.empty(N, dtype=cp.int32) # 只分配 不初始化

index_kernel = cp.RawKernel(kernel_code, 'index_kernel')

# 启动内核：等价于 index_kernel<<<grid, BLOCK>>>(out.data.ptr, N)
index_kernel(grid, (BLOCK, 1, 1), (out, np.int32(N)))

cp.cuda.get_current_stream().synchronize()

expected = cp.arange(N, dtype=cp.int32)
print("grid blocks =", grid[0])
print("总线程数    =", grid[0] * BLOCK)
print("前 5 个     =", out[:5].get())
print("后 5 个     =", out[-5:].get())
print("完全相等 ?  =", bool(cp.array_equal(out, expected)))
print("显存占用 MB =", out.nbytes / 1024 / 1024)