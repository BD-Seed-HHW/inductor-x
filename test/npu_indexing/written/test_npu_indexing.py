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
    y = a + b
    y = y.sum(-1)
    y = y.unsqueeze(3)
    y = y.broadcast_to(Z, Y, X, R) + b
    y = c + y.permute(0, 1, 3, 2)
    return y


@triton.jit
def triton_foo(in_ptr0, in_ptr1, in_ptr2, out_ptr0, BLOCK1: tl.constexpr, BLOCK1_SUB: tl.constexpr,
               BLOCK2: tl.constexpr,
               Z: tl.constexpr, Y: tl.constexpr, X: tl.constexpr, R: tl.constexpr,
               Z_STRIDE: tl.constexpr, Y_STRIDE: tl.constexpr, X_STRIDE: tl.constexpr, R_STRIDE: tl.constexpr,
               Z_STRIDE1: tl.constexpr, Y_STRIDE1: tl.constexpr, X_STRIDE1: tl.constexpr, R_STRIDE1: tl.constexpr,
               ):
    offset: tl.constexpr = tl.program_id(0) * BLOCK1
    base1 = tl.arange(0, BLOCK1_SUB)
    base2 = tl.arange(0, BLOCK2)
    nsub: tl.constexpr = BLOCK1 // BLOCK1_SUB
    # loops1 : tl.constexpr =  nsub * Y * Z
    loops1: tl.constexpr = nsub
    loops2: tl.constexpr = R // BLOCK2

    for z in range(Z):
        for y in range(Y):
            for loop1 in range(loops1):
                # y =  (loop1 // nsub) % Y
                # z = loop1 // nsub // Y
                # off1 = (loop1 % nsub)
                off1 = loop1
                x = offset + (off1 * BLOCK1_SUB) + base1[:, None]
                x1 = offset + (off1 * BLOCK1_SUB) + base1[None, :]
                _tmp4 = tl.full([BLOCK1_SUB, BLOCK2], 0, tl.float32)
                for loop2 in range(loops2):
                    r = loop2 * BLOCK2 + base2[None, :]
                    tmp0 = tl.load(in_ptr0 + (R_STRIDE * r + (X_STRIDE * x) + (Y_STRIDE * y) + (Z_STRIDE * z)), None)
                    tmp1 = tl.load(in_ptr1 + (R_STRIDE * r + (X_STRIDE * x) + (Y_STRIDE * y) + (Z_STRIDE * z)), None)
                    tmp2 = tmp0 + tmp1
                    _tmp4 = _tmp4 + tmp2
                tmp4 = tl.sum(_tmp4, 1)[:, None]
                tmp5 = tmp4.reshape(BLOCK1_SUB, 1).broadcast_to(BLOCK1_SUB, BLOCK2)

                for loop2 in range(loops2):
                    r = loop2 * BLOCK2 + base2[None, :]
                    tmp6 = tl.load(in_ptr1 + (R_STRIDE * r + (X_STRIDE * x) + (Y_STRIDE * y) + (Z_STRIDE * z)), None)
                    tmp7 = tmp6 + tmp5
                    r1 = loop2 * BLOCK2 + base2[:, None]
                    tmp8 = tl.load(in_ptr2 + (R_STRIDE1 * x1 + (X_STRIDE1 * r1) + (Y_STRIDE1 * y) + (Z_STRIDE1 * z)),
                                   None)

                    tmp9 = tmp8.reshape(BLOCK2, BLOCK1_SUB) + tmp7.reshape(BLOCK1_SUB, BLOCK2).permute(1, 0)
                    tl.store(out_ptr0 + (R_STRIDE1 * x1 + (X_STRIDE1 * r1) + (Y_STRIDE1 * y) + (Z_STRIDE1 * z)), tmp9,
                             None)


# target_name = torch.npu.get_device_name(torch.npu.current_device())
guards = {"dummy": None}


def foo_triton_wrapper(a, b, c):
    NBLOCKS = 1
    BLOCK1 = a.shape[2] // NBLOCKS
    BLOCK1_SUB = 64
    BLOCK2 = 64

    compile_opt = ASTSource(
        fn=triton_foo,
        signature={0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32'},
        constants={4: BLOCK1, 5: BLOCK1_SUB, 6: BLOCK2,
                   7: a.shape[0], 8: a.shape[1], 9: a.shape[2], 10: a.shape[3],
                   11: a.stride()[0], 12: a.stride()[1], 13: a.stride()[2], 14: a.stride()[3],
                   15: c.stride()[0], 16: c.stride()[1], 17: c.stride()[2], 18: c.stride()[3],
                   },
        attrs=AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=()),
    )

    # hash_code = "foo_triton_" + str(a.numel()) + "_" + str(b.numel()) + "_" + str(c.numel()) +
    hash_code = "foo_triton_" + str(time.perf_counter())
    compiled_func = guards.get(hash_code)
    if compiled_func == None:
        compiled_func = triton.compile(compile_opt, None, {"debug": True, "mix_mode": "aiv", "name": hash_code})
        guards[hash_code] = compiled_func

    value = torch.empty_strided((c.shape[0], c.shape[1], c.shape[2], c.shape[3]),
                                (c.stride()[0], c.stride()[1], c.stride()[2], c.stride()[3]), dtype=torch.float32).npu()
    compiled_func[NBLOCKS, 1, 1](a, b, c, value)
    return value


if __name__ == "__main__":
    # torch_npu.npu.utils.set_device(1)
    Z, Y, X, R = (1, 1, 64, 64)
    a = torch.randn((Z, Y, X, R), dtype=torch.float32).npu()
    b = torch.randn((Z, Y, X, R), dtype=torch.float32).npu()
    c = torch.randn((Z, Y, R, X), dtype=torch.float32).npu()
    r = foo_triton_wrapper(a, b, c)
    r1 = foo(a, b, c)
    index = 64
    print(r[0, 0, 0:8, 0:8])
    print(r1[0, 0, 0:8, 0:8])
    torch.testing.assert_close(r, r1)
