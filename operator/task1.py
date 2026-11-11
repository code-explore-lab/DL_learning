"""Task 1：Tensor 类骨架

字段与 PyTorch 对齐：
    data         底层存储（先直接复用 numpy，Task 2 再拆出 Storage）
    shape        每个维度的大小
    stride       每个维度跨多少「元素」（不是字节）
    offset       在 storage 里的起始偏移
    requires_grad 是否需要求梯度
    grad         反向传播累积出来的梯度（也是 Tensor）
    grad_fn      生成这个 Tensor 的算子（Function），叶子节点为 None

验收：能创建 Tensor，打印 shape / dtype / device。
"""

import sys

import numpy as np

# 中文 Windows 控制台默认 GBK，输出 emoji 会抛 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _contiguous_stride(shape):
    """按行优先（C order）计算连续布局的 stride。

    例：shape=(2, 3) -> stride=(3, 1)
        shape=(2, 3, 4) -> stride=(12, 4, 1)
    最后一个维度步长为 1，往前每层乘上后一维的大小。
    """
    stride = [0] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        stride[i] = acc
        acc *= shape[i]
    return tuple(stride)


class Tensor:
    """最小可用的 Tensor：只描述「数据 + 形状 + 步长 + 偏移」。

    注意：这里拥有数据的是 numpy ndarray 本身；Tensor 只是一层元信息包装。
    Task 2 会把 data 换成独立的 Storage，让 Tensor 彻底退化成视图描述符。
    """

    def __init__(self, data, dtype=None, requires_grad=False, grad_fn=None):
        # 已经是 ndarray 就复用引用，避免隐式拷贝；list/标量则转换
        arr = np.asarray(data, dtype=dtype)

        self.data = arr
        self.shape = tuple(arr.shape)
        # 新建的 Tensor 默认是连续布局：stride 由 shape 推导，offset 从 0 开始
        self.stride = _contiguous_stride(self.shape)
        self.offset = 0
        self.requires_grad = requires_grad
        self.grad = None          # type: Tensor | None
        self.grad_fn = grad_fn    # type: object | None

    # ---------- 元信息 ----------
    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self):
        # 目前只有 CPU；Task 36 之后再接 CUDA
        return "cpu"

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def numel(self):
        return int(np.prod(self.shape, dtype=np.int64)) if self.shape else 1

    def __len__(self):
        if not self.shape:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    # ---------- 展示 ----------
    def __repr__(self):
        return (
            f"Tensor(shape={self.shape}, dtype={self.dtype}, "
            f"device={self.device}, requires_grad={self.requires_grad})\n"
            f"{self.data}"
        )


def _demo():
    print("=" * 60)
    print("1) 二维整型矩阵")
    a = Tensor(np.arange(6).reshape(2, 3))
    print(a)
    print(f"   stride={a.stride}, offset={a.offset}, numel={a.numel}")

    print("\n2) 浮点 + requires_grad")
    b = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, requires_grad=True)
    print(b)
    print(f"   stride={b.stride}, requires_grad={b.requires_grad}, grad={b.grad}")

    print("\n3) 三维张量")
    c = Tensor(np.zeros((2, 3, 4), dtype=np.float64))
    print(c)
    print(f"   stride={c.stride}, numel={c.numel}")

    print("\n4) 零维标量")
    s = Tensor(3.14)
    print(s)
    print(f"   shape={s.shape}, stride={s.stride}, ndim={s.ndim}")

    print("\n" + "=" * 60)
    print("✅ Task 1 验收通过：Tensor 可创建，并能打印 shape / dtype / device")


if __name__ == "__main__":
    _demo()
