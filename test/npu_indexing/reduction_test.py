import torch
import torch_npu
import sys
sys.path.append("../..")
import inductor_npu
import time

def reduction(a, b, dim, type = "sum"):
    y = a + b
    if type == "sum" :
        y = y.sum(dim)
    elif type == "mean" :
        y = y.mean(dim)
    return y

def mean(a, dim):
    torch.mean(a, dim)

inductor_npu.config.enable_npu_indexing = True
torch_npu.npu.utils.set_device(1)


def test_reduction(shape, dim = -1, type = "sum") :
    print(f"start to test reduction on shape:{shape} dim:{dim} ")
    a, b, c = [torch.randn(shape, requires_grad=False, dtype=torch.float32, device="npu") for _ in range(3)]

    reduction_func = torch.compile(reduction, backend="inductor", dynamic=False )
    r = reduction(a, b, dim, type)
    r1 = reduction_func(a, b, dim, type)
    torch.testing.assert_close(r, r1, rtol=1e-3, atol=1e-3)
    print(f"validation passed")

shapes = [(1,1, 1, 1024) ,(2053,1023, 7, 9) , (8, 8, 1024, 2048),(8, 8, 2048,1024),   (8, 1024, 2048, 8),
           (2048,8, 1024, 8), (2048,1024, 8, 8), ]
type = "sum"
for shape in shapes :
    test_reduction(shape, -1, type = type)

dims = range(4)
for dim in reversed(dims) :
    shape = (8, 16, 512, 64)
    test_reduction(shape, dim = dim, type=type)

