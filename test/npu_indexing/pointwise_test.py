import torch

import sys

import torch_npu
sys.path.append("../..")
import inductor_npu
inductor_npu.config.enable_npu_indexing = True
#torch_npu.npu.utils.set_device(1)

reduction_dim = None
def foo(a, b ):
    y = a + b
    return y


shapes = [ (8, 8, 1024, 2048), (8, 8, 2048,1024),  (8, 1024, 2048, 8), (2048,1024, 8, 8) ,
           (2048,8, 1024, 8), (8, 2048,  8, 1024)]

device = "npu"
def test_pointwise(shape) :
    print(f"start to test pointwise on shape:{shape} ...")
    a, b = [torch.randn(shape, requires_grad=False, dtype=torch.float32, device=device) for _ in range(2)]
    func = torch.compile(foo, backend="inductor")
    r = foo(a, b)
    r1 = func(a, b)
    torch.testing.assert_close(r, r1, rtol=1e-3, atol=1e-3)
    print(f"data validation passed")

for shape in shapes :
    test_pointwise(shape)
