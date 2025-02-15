import torch
import torch_npu
import sys
sys.path.append("../..")
import inductor_npu

def foo(a, b,):
    permute = a.permute(1,0,2).contiguous()
    _,_,bt = torch.split(permute, [2048, 4096, 32], 2)
    clone = bt.contiguous()
    clone_1 = clone.contiguous()

    add = clone_1 + b
    y = torch.exp(add)
    log1p = torch.log1p(y)
    where = torch.where(add < 20, add, log1p)
    return add, where,bt

#torch_npu.npu.utils.set_device(1)

#torch_npu.npu.utils.set_device(2)
a = torch.randn( 4096, 3, 6176, requires_grad=False, dtype=torch.float32, device="npu")
#a = a.permute(1,0,2)
b = torch.randn(32,  requires_grad=False, dtype=torch.float32, device="npu")

compile_foo = torch.compile(foo, backend="inductor")
r,s,_ = foo(a, b)
r1,s1,_ = compile_foo(a, b)
torch.testing.assert_allclose(r, r)

