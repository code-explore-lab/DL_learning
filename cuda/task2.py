import cupy as cp 

kernel_code = r"""
extern "C" __global__ void hello_gpu() {
    int tid = threadIdx.x;
    printf("Hello GPU! threadIdx.x = %d\n", tid);
}
"""
# 把cuad C 编译成 GPU可执行的内核
hello_kernel = cp.RawKernel(kernel_code, "hello_gpu")

grid = (1,1,1)
block = (256, 1, 1)
# 启动Cuda 内核
# 等价原生Cuda C hello_gpu<<<grid, block>>>();
hello_kernel(grid, block, ())

cp.cuda.get_current_stream().synchronize()
