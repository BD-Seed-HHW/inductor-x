import triton
import triton.language as tl
import torch
import torch_npu

@triton.jit
def mamba_kernel(in_ptr0, in_ptr1, in_ptr2, out_ptr0, t0_numel, p1_numel, y2_numel, x3_numel,
                 XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr, RBLOCK: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    base2 = tl.arange(0, RBLOCK)
    loops2: tl.constexpr = (x3_numel + RBLOCK - 1) // RBLOCK
    for t0 in range(t0_numel):
        for loop1 in range(loops1):
            p1_prime = offset + (loop1 * XBLOCK_SUB) + base1[None, :]
            p1 = offset + (loop1 * XBLOCK_SUB) + base1[:, None]
            for y2 in range(y2_numel):
                for loop2 in range(loops2):
                    x3_prime = loop2 * RBLOCK + base2[:, None]
                    x3 = loop2 * RBLOCK + base2[None, :]
                    tmp0 = tl.load(in_ptr0 + (p1_prime + (4096 * x3_prime) + (262144 * y2) + (16777216 * t0)), None)
                    tmp1 = tmp0.permute(1, 0)
                    tmp2 = tl.load(in_ptr1 + (y2 + (32 * p1) + (131072 * t0)), None)
                    tmp3 = tmp2.broadcast_to([XBLOCK_SUB, RBLOCK])
                    tmp6 = tl.load(in_ptr2 + (y2 + (32 * p1) + (131072 * t0)), None)
                    tmp7 = tmp6.broadcast_to([XBLOCK_SUB, RBLOCK])
                    tmp4 = 20.0
                    tmp5 = tmp3 > tmp4
                    tmp8 = tl.where(tmp5, tmp3, tmp7)
                    tmp9 = 0.001
                    tmp10 = tl.maximum(tmp8, tmp9)
                    tmp11 = 0.1
                    tmp12 = tl.minimum(tmp10, tmp11)
                    tmp13 = tmp1 * tmp12
                    tl.store(out_ptr0 + (x3 + (64 * y2) + (2048 * p1) + (8388608 * t0)),
                             tmp13.reshape([XBLOCK_SUB, RBLOCK]), None)


def mamba(in_ptr0, in_ptr1, in_ptr2, t0_numel, p1_numel, y2_numel, x3_numel, stream):
    # The block size of each loop iteration is the smallest power of two greater than the number of columns in 'x'
    num_programs = 32
    XBLOCK = 128
    XBLOCK_SUB = 8
    RBLOCK = 64
    out_ptr0 = torch.empty_strided((3, 4096, 32, 64), (8388608, 2048, 64, 1), device="npu", dtype=torch.float32)
    kernel = mamba_kernel.warmup(in_ptr0, in_ptr1, in_ptr2, out_ptr0, t0_numel, p1_numel, y2_numel, x3_numel,
                                 XBLOCK=XBLOCK, XBLOCK_SUB=XBLOCK_SUB, RBLOCK=RBLOCK, grid=(num_programs,))
    kernel._init_handles()
    # create a number of persistent programs.
    kernel[(num_programs, 1, 1)](in_ptr0, in_ptr1, in_ptr2, out_ptr0, t0_numel, p1_numel, y2_numel, x3_numel,
                                 stream=stream)
    return out_ptr0


device = torch.npu.current_device()
stream = torch.npu.current_stream(device).npu_stream
torch.manual_seed(0)

x = torch.randn(3, 4096, 4096, requires_grad=False, dtype=torch.float32, device="npu")
permute = x.permute(0, 2, 1)
buf11 = torch.split(permute, [2048, 1024, 1024], 2)
in_ptr0 = buf11[0]
buf15 = torch.empty_strided((3, 4096, 32), (131072, 32, 1), device="npu", dtype=torch.float32)
buf16 = torch.empty_strided((3, 4096, 32), (131072, 32, 1), device="npu", dtype=torch.float32)
buf17 = torch.log1p(buf16)
buf18 = buf17

mamba(in_ptr0, buf15, buf18, 3, 4096, 32, 64, stream=stream)
