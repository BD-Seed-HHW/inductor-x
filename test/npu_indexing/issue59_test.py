import torch
import torch_npu
import inductor_npu


from test2.npu_indexing.utils import benchmark_test
def layernorm_backward(x,y,z):
    sum = torch.sum(x)
    mean = sum / torch.numel(sum)
    sub = x - mean
    sqr = sub * sub
    sum_1 = torch.sum(sqr)
    mean_1 = sum_1 / torch.numel(sum_1) + 1e-05
    rsqrt = torch.rsqrt(mean_1)
    mul = sub * rsqrt
    mul_1 = mul * y
    add  = mul_1 + z
    mean_2 = rsqrt / torch.numel(rsqrt)
    return mul, add, mean_2


device = 'npu'

if __name__ == "__main__":
    x = torch.randn((1, 1024), device=device, dtype=torch.float32)
    y = torch.randn((1, 1024), device=device, dtype=torch.float32)
    z = torch.randn((1, 1024), device=device, dtype=torch.float32)


    mul, add, mean_2 = layernorm_backward(x, y, z)
    func = torch.compile(layernorm_backward, backend="inductor", dynamic=False)
    mul_t, add_t, mean_2_t = func(x, y, z)

    torch.testing.assert_close(mul, mul_t, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(add, add_t, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(mean_2, mean_2_t, rtol=1e-3, atol=1e-3)

    print("valid ok")
    benchmark_test(layernorm_backward, func, args=(x, y, z),
                   name="issue59", times=10, repeat=10, profile=False)

