import torch
import torch_npu
import sys
from einops import rearrange

sys.path.append("../..")
import inductor_npu

B,N,S,D = (1, 12, 4096,8) 
def foo(a, b, c ):
    y = a + b
    #y = y.permute(2, 0, 1, 3)
    #y = y.reshape(S,B,N*D)
    y = c + rearrange(y, 'b n s d -> s b (n d)').contiguous()
    
    return y

inductor_npu.config.enable_npu_indexing = True

torch_npu.npu.utils.set_device(1)
a, b, c = [torch.randn(1, 12, 4096,8, requires_grad=False, dtype=torch.float32, device="npu") for _ in range(3)]
d = torch.randn(4096, 1, 96,  requires_grad=False, dtype=torch.float32, device="npu")

func = torch.compile(foo, backend="inductor")

r = func( a, b, d )
r1 = foo (a, b, d )
index=0
#print(r.flatten()[index:index+100])
#print(r1.flatten()[index:index+100])
torch.testing.assert_close(r, r1 , rtol=1e-3, atol=1e-3)



