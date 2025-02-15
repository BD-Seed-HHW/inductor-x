import triton
import triton.language as tl
import torch
import torch_npu

@triton.jit
def var_mean_kernel(in_out_ptr0, in_out_ptr1, in_ptr0, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr,
                    RBLOCK2 : tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    for x0 in range(offset, offset + XBLOCK):
        r1 = tl.arange(0, RBLOCK)[None, :]
        r2 = tl.arange(0, RBLOCK2)[:, None]
        r1_mask = r1 < RBLOCK
        r2_mask = r2 < RBLOCK2
        tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (2048*r2)), r1_mask & r1_mask, other= 0.0)
        tmp1 = tl.reshape(tmp0, [RBLOCK * RBLOCK2])
        # tmp1_1 = tl.reshape( r1_mask & r2_mask, [RBLOCK * RBLOCK2])
        # tmp2 = tl.where(tmp1_1, tmp1,  0)
        tmp3 = tl.sum(tmp1, 0)
        tmp4 = 512.0
        tmp5 = tmp3 / tmp4
        tmp6 = tmp0 - tmp5
        tmp7 = tmp6 * tmp6
        tmp8 = tl.reshape(tmp7,  [RBLOCK * RBLOCK2])
        tmp10 = tl.sum(tmp8, 0)
        tmp11 = 511.0
        tmp12 = tmp10 / tmp11
        tl.store(in_out_ptr0 + x0 + tl.arange(0, 1), tmp12, None)
        tl.store(in_out_ptr1 + x0 + tl.arange(0, 1), tmp5, None)


def var_mean(x, stream):
    X, Y, Z = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    num_programs = 32
    RBLOCK = triton.next_power_of_2(Z)
    RBLOCK2 = triton.next_power_of_2(X)

    num_programs = num_programs if Y > num_programs else 1
    XBLOCK = (Y + num_programs -1) // num_programs

    mean = torch.empty_strided((Y,), (1,), device='npu', dtype=torch.float32)
    var = torch.empty_strided((Y,), (1,), device='npu', dtype=torch.float32)

    kernel = var_mean_kernel.warmup(var, mean, x,  XBLOCK=XBLOCK, RBLOCK = RBLOCK, RBLOCK2=RBLOCK2, grid=(num_programs,))
    kernel._init_handles()

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](var, mean, x,  stream=stream )
    return (var, mean)

device = torch.npu.current_device()
stream = torch.npu.current_stream(device).npu_stream
torch.manual_seed(0)


x = torch.randn(8,32,64, device='npu')
v,m = var_mean(x, stream)
v1, m1 = torch.var_mean(x, [0,2])
# print(m1, m)
# print(v1, v)
assert torch.allclose(v, v1)