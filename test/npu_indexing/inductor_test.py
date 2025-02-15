import torch
import torch_npu
import sys
sys.path.append("../..")
import inductor_npu


def foo(a, d ,shape ):
    y = a.reshape(shape)
    y = y.permute(0,2,1) + d
    return y

torch.npu.utils.set_device(0)
# shapes = [(513,512,256)]

shapes = [(8,2048,4), (8,2048,3),   (8,526,3), (50,526,3),(50,526,129),]
for shape in shapes:
    print(f"start data validation on shape:{shape}")
    a = torch.randn(shape[0],shape[1] * shape[2], requires_grad=False, dtype=torch.float32, device="npu")
    d =  torch.randn(shape[0],shape[2], shape[1],  requires_grad=False, dtype=torch.float32, device="npu")

    func = torch.compile(foo, backend="inductor", dynamic=False)

    r = func( a,  d, shape)
    r1 = foo (a, d ,shape )
    torch.testing.assert_close(r, r1 , rtol=1e-3, atol=1e-3)
    print(f"data validation passed")


