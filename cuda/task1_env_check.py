import subprocess
import sys

# 控制台/管道在中文 Windows 下默认是 GBK，输出 emoji 会抛 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def check_torch():
    print("===check Pytorch CUDA====")
    try:
        import torch
        print(f"torch version: {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"torch.cuda.is_available() == {cuda_ok}")
        if cuda_ok:
            dev_cnt = torch.cuda.device_count()
            print(f"GPU count: {dev_cnt}")
            for i in range(dev_cnt):
                print(f"GPU[{i}]: {torch.cuda.get_device_name(i)}")
        return cuda_ok
    except Exception as e:
        print(f"Torch check Failed: {e}")
        return False
def check_cupy():
    print("\n===== Check CuPy =====")
    try:
        import cupy
        print(f"cupy version: {cupy.__version__}")
        # 查设备名（cupy 14.x 正确 API）
        props = cupy.cuda.runtime.getDeviceProperties(0)
        print(f"cupy device: {props['name'].decode()}")
        # 跑一个 GPU matmul 验证真的能用
        a = cupy.random.rand(1024, 1024, dtype=cupy.float32)
        b = a @ a
        cupy.cuda.Stream.null.synchronize()
        print(f"cupy 1024x1024 matmul OK, sum={float(b.sum()):.1f}")
        print("cupy CUDA available: OK")
        return True
    except Exception as e:
        print(f"CuPy check FAILED: {e}")
        return False

def check_nvcc():
    print("\n===== Check nvcc --version =====")
    try:
        res = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(res.stdout)
        return True
    except Exception as e:
        print(f"nvcc check FAILED: {e}")
        return False

if __name__ == "__main__":
    r1 = check_torch()
    r2 = check_cupy()
    r3 = check_nvcc()

    print("\n===== Summary =====")
    print(f"Torch CUDA: {'✅ PASS' if r1 else '❌ FAIL'}")
    print(f"CuPy:       {'✅ PASS' if r2 else '❌ FAIL'}")
    print(f"nvcc:       {'✅ PASS' if r3 else '❌ FAIL'}")
    if r1 and r2 and r3:
        print("\n🎉 All three checks passed! Found RTX3060.")
    else:
        print("\n⚠️ Some checks failed, check above log.")
        sys.exit(1)