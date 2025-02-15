import torch
import torch_npu
import sys
sys.path.append("../..")
import inductor_npu
import time

def var_mean(a,  dim):
    y = torch.var_mean(a, dim)
    return y

torch_npu.npu.utils.set_device(1)
device = "npu"
def test_var_mean(shape, dim) :
    print(f"start to test var_mean on shape:{shape} dim:{dim} ")
    a = torch.randn(shape, requires_grad=False, dtype=torch.float32, device=device)

    var_mean_triton = torch.compile(var_mean, backend="inductor", dynamic=False)
    r = var_mean(a, dim)
    r1 = var_mean_triton(a, dim)
    torch.testing.assert_close(r[0], r1[0], rtol=1e-3, atol=1e-3)
    print("mean validation passed")
    torch.testing.assert_close(r[1], r1[1], rtol=1e-3, atol=1e-3)
    print("var validation passed")



shape = (128,8,64)
dims = [[0,2], [0,1], [1,2], [0], [1], [2]]
for dim in dims:
    test_var_mean(shape, dim)
shapes = [(1,512), (1,1024)]
dims = [[1]]
for shape in shapes:
    for dim in dims:
        test_var_mean(shape, dim)
# #(1,2,1024),
# shapes = [ (1,4,512), (256,8,1), (256,1,4)]
# #[0,2]
# dims = [ [0,1], [1,2], [0], [1], [2]]
# for shape in shapes :
#     for dim in dims:
#         test_var_mean(shape, dim)

