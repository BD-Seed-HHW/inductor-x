import torch
import torch_npu
import sys
sys.path.append("../..")
import inductor_npu

def foo(a, b, c , shape):
    y = a + b
    y = c + y.permute(shape)
    return y
inductor_npu.config.enable_npu_indexing = True
#torch_npu.npu.utils.set_device(1)

#torch_npu.npu.utils.set_device(2)
a, b, = [torch.randn(8, 8, 512,128, requires_grad=False, dtype=torch.float32, device="npu") for _ in range(2)]
d = torch.randn(8, 8, 512,128,  requires_grad=False, dtype=torch.float32, device="npu")
0,1,2,3

shapes = [(2,0,1,3),(1,0,2,3),(1,0,2,3),(0,1,3,2), (3,0,1,2), (0,2,1,3)]
for shape in shapes :
    print(f"start to test permute on shape :{shape} ")
    c = d.permute(shape).contiguous()
    func = torch.compile(foo, backend="inductor")
    r = func( a, b, c , shape)
    r1 = foo (a, b, c , shape)
    index=100
    #print(r[0,0,index:index+64])
    #print(r1[0,0,index:index+64])
    torch.testing.assert_close(r, r1 , rtol=1e-3, atol=1e-3)
    print("data validation passed")
  #  break


