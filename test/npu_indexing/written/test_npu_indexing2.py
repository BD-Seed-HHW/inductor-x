# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import torch
import torch_npu
import triton
import triton.language as tl
from triton.compiler import ASTSource, AttrsDescriptor
import time


# (2, 2, 2048, 1024) (2, 2, 2048, 1024)   ok
# (2,  2048, 2,  1024)  (2, 1024 , 2,  2048)  ok


def foo(a, b, c):
    y = a + b + c
    y = y.sum(dim=1)
    #    y = y.unsqueeze(3)
    #    y = y.broadcast_to(Z, Y, X, R) + b
    #    y = c + y.permute(0, 1, 3, 2)
    return y


@triton.jit
def triton_codegen2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr,
                    RBLOCK: tl.constexpr):
    ynumel = 8
    rnumel = 2048
    xnumel = 1024
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
    base2 = tl.arange(0, RBLOCK)
    loops2: tl.constexpr = rnumel // RBLOCK
    for y in range(ynumel):
        y0 = y
        for loop1 in range(loops1):
            x = offset + (loop1 * XBLOCK_SUB) + base1
            x1 = offset + (loop1 * XBLOCK_SUB) + base1[None, :]
            _tmp6 = tl.full([XBLOCK_SUB, RBLOCK], 0, tl.float32)
            for loop2 in range(loops2):
                r2 = loop2 * RBLOCK + base2[:, None]
                tmp0 = tl.load(in_ptr0 + (x1 + (1024 * r2) + (2097152 * y0)), None, eviction_policy='evict_last')
                tmp1 = tl.load(in_ptr1 + (x1 + (1024 * r2) + (2097152 * y0)), None, eviction_policy='evict_last')
                tmp3 = tl.load(in_ptr2 + (x1 + (1024 * r2) + (2097152 * y0)), None, eviction_policy='evict_last')
                tmp2 = tmp0 + tmp1
                tmp4 = tmp2 + tmp3
                tmp5 = tl.reshape(tmp4, [RBLOCK, XBLOCK_SUB])
                tmp7 = _tmp6 + tmp5
                _tmp6 = tmp7
            tmp6 = tl.sum(_tmp6, 0).reshape(XBLOCK_SUB)

            tl.store(out_ptr0 + (x + (1024 * y0)), tmp6, None)


# target_name = torch.npu.get_device_name(torch.npu.current_device())
guards = {"dummy": None}


def foo_triton_wrapper(a, b, c):
    NBLOCKS = 8
    BLOCK1 = a.shape[2] // NBLOCKS
    BLOCK1_SUB = 64
    BLOCK2 = 64

    compile_opt = ASTSource(
        fn=triton_codegen2,
        signature={0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32'},
        constants={4: BLOCK1, 5: BLOCK1_SUB, 6: BLOCK2},
        attrs=AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=()),
    )

    # hash_code = "foo_triton_" + str(a.numel()) + "_" + str(b.numel()) + "_" + str(c.numel()) +
    hash_code = "foo_triton_" + str(time.perf_counter())
    compiled_func = guards.get(hash_code)
    if compiled_func == None:
        compiled_func = triton.compile(compile_opt, None, {"debug": True, "mix_mode": "aiv", "name": hash_code})
        guards[hash_code] = compiled_func

    value = torch.empty_strided((c.shape[0], c.shape[2]),
                                (c.shape[2], 1), dtype=torch.float32).npu()
    compiled_func[NBLOCKS, 1, 1](a, b, c, value)
    return value


if __name__ == "__main__":
    # torch_npu.npu.utils.set_device(1)
    Y, X, R = (8, 2048, 1024)
    a = torch.randn((Y, X, R), dtype=torch.float32).npu()
    b = torch.randn((Y, X, R), dtype=torch.float32).npu()
    c = torch.randn((Y, X, R), dtype=torch.float32).npu()
    r = foo_triton_wrapper(a, b, c)
    r1 = foo(a, b, c)
    print(r[0:8, 0:8, ])
    print(r1[0:8, 0:8])
    torch.testing.assert_close(r, r1, rtol=1e-3, atol=1e-3)
