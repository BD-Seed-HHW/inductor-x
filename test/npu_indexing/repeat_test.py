import torch
import torch_npu
import sys
sys.path.append("../..")
import inductor_npu
import time

def repeat(a,  dim):
    y = a.repeat(*dim)
    return y

torch_npu.npu.utils.set_device(1)
device = "npu"
def test_repeat(shape, dim) :
    print(f"start to test repeat on shape:{shape} dim:{dim} ")
    a = torch.randn(shape, requires_grad=False, dtype=torch.float32, device=device)

    repeat_triton = torch.compile(repeat, backend="inductor")
    r = repeat(a, dim=dim)
    r1 = repeat_triton(a, dim=dim)
    torch.testing.assert_close(r, r1, rtol=1e-3, atol=1e-3)
    print("validation passed")




#test_repeat((8,32,64), dim=(1,1,2))
test_repeat((8,1024,64), dim=(1,2,1))
test_repeat((8,1024,64), dim=(2,1,1))
test_repeat((8,1024,64), dim=(1,2,2))
test_repeat((8,1024,64), dim=(2,2,2))