import torch
import torch_npu
import sys
sys.path.append("../..")
import inductor_npu

reduction_dim = None

shape = (8, 8, 256,128)
def foo(a, b, c , dim, permute_shape):
    y = a + b
    y = y.sum(dim)
    y = y.unsqueeze(dim)
    y = y.broadcast_to(shape) + b
    y = c + y.permute(permute_shape)
    return y

inductor_npu.config.enable_npu_indexing = True
dims = 4
permuted_shapes = [(0,1,3,2),(2,0,1,3),(1,0,2,3), (3,0,1,2), (0,2,1,3)]
#permuted_shapes = [(1,0,2,3),]
a, b = [torch.randn(shape, requires_grad=False, dtype=torch.float32, device="npu") for _ in range(2)]
d = torch.randn(shape,  requires_grad=False, dtype=torch.float32, device="npu")

for dim in [3,2,1,0] :
    for permute_shape in permuted_shapes :
        print(f"start reduction_broadcast_permute on dim:{dim}, shape:{shape} permute_shape:{permute_shape} ")
        c = d.permute(permute_shape).contiguous()
        func = torch.compile(foo, backend="inductor")
        r = func( a, b, c, dim , permute_shape)
        r1 = foo (a, b, c, dim , permute_shape)
        torch.testing.assert_close(r, r1 , rtol=1e-3, atol=1e-3)
        print(f"data validation passed")