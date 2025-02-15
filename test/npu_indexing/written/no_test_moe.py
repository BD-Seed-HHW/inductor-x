import time

import triton
import triton.language as tl
from triton.compiler import ASTSource, AttrsDescriptor
import torch
import torch_npu


@triton.jit
def triton_moe_forward(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, y0_numel, x1_numel,
            XBLOCK : tl.constexpr, XBLOCK_SUB : tl.constexpr, RBLOCK : tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    base2 = tl.arange(0, RBLOCK)
    loops2: tl.constexpr = x1_numel // RBLOCK
    for loop1 in range(loops1):
        y0_prime = offset + (loop1 * XBLOCK_SUB) + base1[None, :]
        y0 = offset + (loop1 * XBLOCK_SUB) + base1[:, None]
        for loop2 in range(loops2):
            x1_prime = loop2 * RBLOCK + base2[:, None]
            x1 = loop2 * RBLOCK + base2[None, :]
            tmp0 = tl.load(in_ptr0 + (x1 + (16*y0)), None)
            tmp5 = tl.load(in_ptr1 + (x1 + (8*y0)), None)
            tmp8 = tl.load(in_ptr2 + (0))
            #tmp8 = tmp5
            tmp9 = tl.broadcast_to(tmp8, [XBLOCK_SUB, RBLOCK])
            tmp13 = tl.load(in_ptr0 + (8 + x1 + (16*y0)), None)
            tmp15 = tl.load(in_ptr3 + (x1 + (8*y0)), None)
            tmp18 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
            tmp19 = tmp18.broadcast_to([XBLOCK_SUB, RBLOCK])
            tmp1 = tmp0.to(tl.float32)
            tmp2 = tl.full([1, 1], 1, tl.int32)
            tmp3 = tl.full([1, 1], 0, tl.int32)
            tmp4 = tmp2 == tmp3
            tmp6 = tmp5 - tmp2
            tmp7 = tmp6.to(tl.int64)
            tmp10 = tl.where(tmp7.to(tl.float32) < tmp9.to(tl.float32), 1.0, 0.0)
            tmp11 = tmp10.to(tl.int64)
            tmp12 = tmp0 * tmp11
            tmp14 = tl.where(tmp4, tmp12, tmp13)
            tmp16 = tmp15 - tmp2
            tmp17 = tmp16.to(tl.float32)
            tmp20 = tmp17 + tmp19
            tmp21 = tmp9.to(tl.float32)
            tmp22 = tl.where(tmp20.to(tl.float32) < tmp21.to(tl.float32), 1.0, 0.0)
            tmp23 = tmp22.to(tl.int64)
            tmp24 = tmp14 * tmp23
            tl.store(out_ptr0 + (x1 + (8 * y0)), tmp1, None)
            tl.store(out_ptr1 + (x1 + (8 * y0)), tmp24, None)


guards = {"dummy": None}

def moe_forward_triton(arg0_1, buf13, arg2_1, buf15, buf16):
    XBLOCK,XBLOCK_SUB,RBLOCK =(128, 64, 8 )
    NBLOCKS = 32
    compile_opt = ASTSource(
        fn=triton_moe_forward,
        signature={0: '*i64', 1: '*i32', 2: '*i64', 3: '*i32', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: 'i32', 8: 'i32'},
        constants={9: XBLOCK, 10: XBLOCK_SUB, 11: RBLOCK,},
        attrs=AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=()),
    )

    hash_code = "moe_forward_triton_" + str(time.perf_counter())
    compiled_func = guards.get(hash_code)
    if compiled_func == None:
        compiled_func = triton.compile(compile_opt, None,
                                       {"debug": True, "mix_mode": "aiv", "name": hash_code})
        guards[hash_code] = compiled_func

    buf17 = torch.empty_strided((4096, 8), (8, 1), device='npu', dtype=torch.int64)
    buf2 = torch.empty_strided((4096, 8), (8, 1), device='npu', dtype=torch.float32)
    compiled_func[NBLOCKS, 1, 1](arg0_1, buf13, arg2_1, buf15, buf16, buf2, buf17, 4096, 8)

    return  buf2, buf17
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor

if __name__ == "__main__" :
    arg0_1 = torch.empty_strided((4096, 2, 8), (16, 8, 1), device='npu', dtype=torch.int64)
    arg1_1 = torch.empty_strided((4096, 8), (8, 1), device='npu', dtype=torch.float32)
    arg2_1 = torch.empty_strided((1,), (1,), device='npu', dtype=torch.int64)
    #arg0_1, buf13, arg2_1, buf15, buf16, buf2, buf17, 4096, 8, grid = grid(4096, 8),
    buf13 = torch.cumsum(reinterpret_tensor(arg0_1, (4096, 8), (16, 1), 0), dim=0, dtype=torch.int32)
    buf15 = torch.cumsum(reinterpret_tensor(arg0_1, (4096, 8), (16, 1), 8), dim=0, dtype=torch.int32)
    buf16 = torch.empty_strided((1, 8), (8, 1), device='npu', dtype=torch.float32 )
    buf2,buf17 = moe_forward_triton(arg0_1, buf13, arg2_1, buf15, buf16)



