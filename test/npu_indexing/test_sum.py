import time

import torch
import torch_npu
import triton
import triton.language as tl
from triton.compiler import ASTSource, AttrsDescriptor
import time
@triton.jit
def test_sum_double_loop(in_ptr0, in_ptr1, in_ptr2, out_ptr0, rnumel, xnumel,
                  XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr, RBLOCK: tl.constexpr):
    R = rnumel
    X = xnumel
    xoffset = tl.program_id(0) * XBLOCK
    xbase = xoffset + tl.arange(0, XBLOCK_SUB)
    rbase = tl.arange(0, RBLOCK)
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset_sub + xbase
        x0 = xindex[None, :]
        _tmp6 = tl.full([RBLOCK, XBLOCK_SUB], 0, tl.float32)
        for roffset in range(0, rnumel, RBLOCK):
            rindex = roffset + rbase
            rmask = None
            r1 = rindex[:, None]
            tmp0 = tl.load(in_ptr0 + (X * r1 + (x0)), rmask)
            tmp1 = tl.load(in_ptr1 + (X * r1 + (x0)), rmask)
            tmp3 = tl.load(in_ptr2 + (X * r1 + (x0)), rmask)
            tmp2 = tmp0 + tmp1
            tmp4 = tmp2 + tmp3
            _tmp6 = _tmp6 + tmp4
        tmp6 = tl.sum(_tmp6, 0)
        tl.store(out_ptr0 + (xindex), tmp6, None)


@triton.jit
def persistent_sum_0(in_ptr0, in_ptr1, out_ptr0, r0_numel, ):
    RBLOCK: tl.constexpr = 1024
    r0 = tl.arange(0, RBLOCK)
    r0_mask = r0 < r0_numel
    tmp0 = tl.load(in_ptr0 + (r0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0), r0_mask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.reshape(tmp2, [RBLOCK])
    tmp5 = tl.where(r0_mask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 0)
    tl.store(out_ptr0 + (0), tmp6, None)
# def persistent_sum_0(in_ptr0, in_ptr1, out_ptr0, r0_numel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
#     RBLOCK: tl.constexpr = 1024
#     r0_prime = tl.arange(0, RBLOCK)
#     r0 = tl.arange(0, RBLOCK)
#     r0_mask = r0 < r0_numel
#     r0_prime_mask = r0_prime < r0_numel
#     tmp0 = tl.load(in_ptr0 + (r0), r0_mask, other=0.0)
#     tmp1 = tl.load(in_ptr1 + (r0), r0_mask, other=0.0)
#     tmp2 = tmp0 + tmp1
#     tmp3 = tl.reshape(tmp2, [RBLOCK])
#     tmp5 = tl.where(r0_mask, tmp3, 0)
#     tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
#     tl.store(out_ptr0 + (tl.arange(0,1) ), tmp6, None)


#target_name = torch.npu.get_device_name(torch.npu.current_device())
guards = {"dummy":None}
def test_sum_triton(a, b, c):
    NBLOCKS = 32
    XBLOCK = a.shape[1] // NBLOCKS
    XBLOCK_SUB = 64
    RBLOCK = 64

    hash_code = "tt_test_sum_" + str(time.perf_counter())

    value = torch.empty_strided((a.shape[1],), (1,)).npu()
    compiled_func = guards.get(hash_code)
    if compiled_func == None :
        compiled_func = test_sum_double_loop.warmup( a,b,c,value, a.shape[0], a.shape[1],
                                       XBLOCK=XBLOCK,XBLOCK_SUB=XBLOCK_SUB, RBLOCK =RBLOCK, grid=(NBLOCKS,))
        guards[hash_code] = compiled_func

    device = torch.npu.current_device()
    stream = torch.npu.current_stream(device).npu_stream
    torch.manual_seed(0)
    compiled_func[NBLOCKS, 1, 1](a,b,c,value, a.shape[0], a.shape[1], stream = stream)
    return value

def test_persitent_sum_triton(a,b,c):
    NBLOCKS = 1

    hash_code = "test_persitent_sum_" + str(time.perf_counter())


    compiled_func = guards.get(hash_code)
    if compiled_func == None :
        compiled_func = persistent_sum_0.warmup( a,b,c, a.shape[1],
                                       grid=(NBLOCKS,))
        guards[hash_code] = compiled_func

    device = torch.npu.current_device()
    stream = torch.npu.current_stream(device).npu_stream
    torch.manual_seed(0)
    compiled_func[NBLOCKS, 1, 1](a,b,c, a.shape[1], stream = stream)
    return c

def foo(a, b, c):
    y = a + b + c
    y = y.sum(0)
    return y

def bar(a, b, c):
    y = a + b + c
    y = y.sum(1)
    return y

X = 64
Y =  1024*1024
#torch_npu.npu.utils.set_device(0)
if __name__ == "__main__":
    torch.npu.utils.set_device(1)
    x = torch.randn((1, 1024), dtype = torch.float32).npu()
    y = torch.randn((1, 1024), dtype = torch.float32).npu()
    z = torch.randn((1, 1), dtype=torch.float32).npu()
    value = test_persitent_sum_triton(x,y,z)

    # a = torch.randn((X, Y), dtype = torch.float32).npu()
    # b = torch.randn((X, Y), dtype = torch.float32).npu()
    # c = torch.randn((X, Y), dtype = torch.float32).npu()
    #
    # # r = test_sum_triton_low(a, b, c)
    # # r1 = bar(a, b, c)
    # # torch.testing.assert_close(r, r1, rtol=1e-3, atol=1e-3)
    # # print("reduction passed")
    # r = test_sum_triton(a, b, c)
    # r1 = foo(a, b, c)
    # # print(r[0:8])
    # # print(r1[0:8])
    # torch.testing.assert_close(r, r1, rtol=1e-3, atol=1e-3)
    # print("high-order reduction passed")




