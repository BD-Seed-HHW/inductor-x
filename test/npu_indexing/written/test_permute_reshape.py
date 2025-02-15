# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import torch
import torch_npu
import triton
import triton.language as tl
from triton.compiler import ASTSource, AttrsDescriptor
import time


@triton.jit
def triton_foo(in_ptr0, in_ptr1, in_ptr2, out_ptr0, BLOCK1: tl.constexpr, BLOCK1_SUB: tl.constexpr,
               BLOCK2: tl.constexpr, S: tl.constexpr, N: tl.constexpr, D: tl.constexpr
               ):
    offset: tl.constexpr = tl.program_id(0) * BLOCK1
    base1 = tl.arange(0, BLOCK1_SUB)
    base2 = tl.arange(0, BLOCK2)
    loops1: tl.constexpr = BLOCK1 // BLOCK1_SUB
    loops2: tl.constexpr = D // BLOCK2

    for loop1 in range(loops1):
        off1 = loop1
        s = offset + (off1 * BLOCK1_SUB) + base1[:, None]
        for n in range(N):
            for loop2 in range(loops2):
                d = loop2 * BLOCK2 + base2[None, :]
                tmp0 = tl.load(in_ptr0 + ((32768 * n) + (8 * s) + d), None)
                tmp1 = tl.load(in_ptr1 + ((32768 * n) + (8 * s) + d), None)
                tmp2 = tmp0 + tmp1

                tmp3 = tl.load(in_ptr2 + ((8 * n) + d + (96 * s)), None)
                tmp9 = tmp3 + tmp2
                tl.store(out_ptr0 + ((8 * n) + d + (96 * s)), tmp9, None)


# target_name = torch.npu.get_device_name(torch.npu.current_device())
guards = {"dummy": None}


def foo_triton_wrapper(a, b, c):
    NBLOCKS = 32 if a.shape[2] >= 256 else 1
    BLOCK1 = a.shape[2] // NBLOCKS
    BLOCK1_SUB = BLOCK1 if BLOCK1 < 64 else 64
    BLOCK2 = a.shape[3] if a.shape[3] < 64 else 64

    compile_opt = ASTSource(
        fn=triton_foo,
        signature={0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32'},
        constants={4: BLOCK1, 5: BLOCK1_SUB, 6: BLOCK2,
                   7: a.shape[2], 8: a.shape[1], 9: a.shape[3]
                   },
        attrs=AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=()),
    )

    # hash_code = "foo_triton_" + str(a.numel()) + "_" + str(b.numel()) + "_" + str(c.numel()) +
    hash_code = "foo_triton_" + str(time.perf_counter())
    compiled_func = guards.get(hash_code)
    if compiled_func == None:
        compiled_func = triton.compile(compile_opt, None, {"debug": True, "mix_mode": "aiv", "name": hash_code})
        guards[hash_code] = compiled_func

    value = torch.empty_strided((c.shape[0], c.shape[1], c.shape[2]),
                                (c.stride()[0], c.stride()[1], c.stride()[2]), dtype=torch.float32).npu()
    compiled_func[NBLOCKS, 1, 1](a, b, c, value)
    return value


B, N, S, D = (1, 12, 4096, 8)
from einops import rearrange


def foo(a, b, c):
    y = a + b
    # y = c + rearrange(y, 'b n s d -> s b (n d)').contiguous()
    y = c + y.permute(2, 0, 1, 3).reshape(S, B, N * D)
    return y


if __name__ == "__main__":
    # torch_npu.npu.utils.set_device(1)

    a = torch.randn((B, N, S, D), dtype=torch.float32).npu()
    b = torch.randn((B, N, S, D), dtype=torch.float32).npu()
    c = torch.randn((S, B, N * D), dtype=torch.float32).npu()
    r = foo_triton_wrapper(a, b, c)
    r1 = foo(a, b, c)
    print(r[0:8, 0, 0:8])
    print(r1[0:8, 0, 0:8])
    torch.testing.assert_close(r, r1, rtol=1e-3, atol=1e-3)
    print("data validation passed")
