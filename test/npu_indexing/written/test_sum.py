# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

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
def test_sum_loop_low(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, ynumel,
                      XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    X = xnumel
    Y = ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)

    x0 = xindex[:, None]
    rbase = tl.arange(0, RBLOCK)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, ynumel, RBLOCK):
        rindex = roffset + rbase
        rmask = None
        r1 = rindex[None, :]
        tmp0 = tl.load(in_ptr0 + (r1 + (Y * x0)), rmask)
        tmp1 = tl.load(in_ptr1 + (r1 + (Y * x0)), rmask)
        tmp3 = tl.load(in_ptr2 + (r1 + (Y * x0)), rmask)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        _tmp6 = _tmp6 + tmp4
    tmp6 = tl.sum(_tmp6, 1)

    tl.store(out_ptr0 + (xindex), tmp6, None)


# target_name = torch.npu.get_device_name(torch.npu.current_device())
guards = {"dummy": None}


def test_sum_triton(a, b, c):
    NBLOCKS = 32
    XBLOCK = a.shape[1] // NBLOCKS
    XBLOCK_SUB = 64
    RBLOCK = 64

    compile_opt = ASTSource(
        fn=test_sum_double_loop,
        signature={0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'},
        constants={6: XBLOCK, 7: XBLOCK_SUB, 8: RBLOCK},
        attrs=AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=()),
    )

    hash_code = "tt_test_sum_" + str(time.perf_counter())

    compiled_func = guards.get(hash_code)
    if compiled_func == None:
        compiled_func = triton.compile(compile_opt, None, {"debug": True, "mix_mode": "aiv", "name": hash_code})
        guards[hash_code] = compiled_func

    value = torch.empty_strided((a.shape[1],), (1,)).npu()
    compiled_func[NBLOCKS, 1, 1](a, b, c, value, a.shape[0], a.shape[1])
    return value


def test_sum_triton_low(a, b, c):
    XBLOCK = 64
    RBLOCK = 32
    NBLOCKS = a.shape[0] // XBLOCK

    compile_opt = ASTSource(
        fn=test_sum_loop_low,
        signature={0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'},
        constants={6: XBLOCK, 7: RBLOCK},
        attrs=AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=()),
    )

    hash_code = "tt_test_sum_" + str(time.perf_counter())

    compiled_func = guards.get(hash_code)
    if compiled_func == None:
        compiled_func = triton.compile(compile_opt, None, {"debug": True, "mix_mode": "aiv", "name": hash_code})
        guards[hash_code] = compiled_func

    value = torch.empty_strided((a.shape[0],), (1,)).npu()
    compiled_func[NBLOCKS, 1, 1](a, b, c, value, a.shape[0], a.shape[1])
    return value


def foo(a, b, c):
    y = a + b + c
    y = y.sum(0)
    return y


def bar(a, b, c):
    y = a + b + c
    y = y.sum(1)
    return y


X = 64
Y = 1024 * 1024
# torch_npu.npu.utils.set_device(0)
if __name__ == "__main__":
    torch.npu.utils.set_device(1)
    a = torch.randn((X, Y), dtype=torch.float32).npu()
    b = torch.randn((X, Y), dtype=torch.float32).npu()
    c = torch.randn((X, Y), dtype=torch.float32).npu()

    # r = test_sum_triton_low(a, b, c)
    # r1 = bar(a, b, c)
    # torch.testing.assert_close(r, r1, rtol=1e-3, atol=1e-3)
    # print("reduction passed")
    r = test_sum_triton(a, b, c)
    r1 = foo(a, b, c)
    # print(r[0:8])
    # print(r1[0:8])
    torch.testing.assert_close(r, r1, rtol=1e-3, atol=1e-3)
    print("high-order reduction passed")
