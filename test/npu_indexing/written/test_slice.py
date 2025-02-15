# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import triton
import triton.language as tl
from triton.compiler import ASTSource, AttrsDescriptor

npu_target = triton.runtime.driver.active.get_current_target("npu")

import torch
import torch_npu


@triton.jit
def fn_slice_clone(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)  # [:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 0 + (32 * x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)


ast_slice_clone = ASTSource(
    fn=fn_slice_clone,
    signature={0: '*fp32', 1: '*fp32', 2: 'i32'},
    constants={3: 16},
    attrs=AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=()),
)


@triton.jit
def fn_slice_assign(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + 2 + (32 * x0), tmp0, xmask)


ast_slice_assign = ASTSource(
    fn=fn_slice_assign,
    signature={0: '*fp16', 1: '*fp16', 2: 'i32'},
    constants={3: 16},
    attrs=AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=()),
)


@triton.jit
def fn_slice_update(in_ptr0, in_out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + 3 + (32 * x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + 3 + (32 * x0), tmp2, xmask)


ast_slice_update = ASTSource(
    fn=fn_slice_update,
    signature={0: '*fp16', 1: '*fp16', 2: 'i32'},
    constants={3: 16},
    attrs=AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=()),
)


def test_slice_clone():
    ret = triton.compile(ast_slice_clone, npu_target, {"debug": True, "mix_mode": "aiv"})
    x = torch.randn((16, 32), dtype=torch.float32).npu()
    out = torch.randn((16,), dtype=torch.float32).npu()
    ret[1, 1, 1](x, out, 16)
    assert (torch.equal(out, x[..., 0]))
    print("test_slice_clone passed")


def test_slice_assign():
    ret = triton.compile(ast_slice_assign, npu_target, {"debug": True, "mix_mode": "aiv"})
    x = torch.randn((16,), dtype=torch.float16).npu()
    out = torch.randn((16, 32), dtype=torch.float16).npu()
    y = out.clone()
    y[..., 2] = x
    ret[1, 1, 1](x, out, 16)
    assert (torch.equal(out, y))
    print("test_slice_assign passed")


def test_slice_update():
    ret = triton.compile(ast_slice_update, npu_target, {"debug": True, "mix_mode": "aiv"})
    x = torch.randn((16,), dtype=torch.float16).npu()
    out = torch.randn((16, 32), dtype=torch.float16).npu()
    y = out.clone()
    y[..., 3] += x
    ret[1, 1, 1](x, out, 16)
    assert (torch.equal(out, y))
    print("test_slice_update passed")


from triton.language.extra.helpers import slice_at_index


@triton.jit
def fn_cache_slice(in_ptr0, out_ptr0, out_ptr1):
    xindex = tl.arange(0, 8)[None, :]
    yindex = tl.arange(0, 16)[:, None]
    tmp0 = tl.load(in_ptr0 + (xindex) + (yindex * 8), None, eviction_policy='evict_last')
    tmp1 = tmp0 * 2
    tl.store(out_ptr0 + (xindex) + (yindex * 8), tmp1, None, eviction_policy="evict_last")
    tmp4 = slice_at_index(tmp1, dim=0, offset=4)
    yindex2 = tl.arange(0, 8)
    tl.store(out_ptr1 + yindex2, tmp4, None, eviction_policy="evict_last")


ast_cache_slice = ASTSource(
    fn=fn_cache_slice,
    signature={0: '*fp32', 1: '*fp32', 2: '*fp32'},
    constants={},
    attrs=AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=()),
)


def test_cache_slice():
    ret = triton.compile(ast_cache_slice, npu_target, {"debug": True, "mix_mode": "aiv"})
    x = torch.randn((16, 8), dtype=torch.float32).npu()
    out0 = torch.randn((16, 8), dtype=torch.float32).npu()
    out1 = torch.randn((8,), dtype=torch.float32).npu()
    ret[1, 1, 1](x, out0, out1)
    assert (torch.equal(out0, x * 2) and torch.equal(out1, x[4, ...] * 2))


if __name__ == "__main__":
    test_cache_slice()
    test_slice_clone()
    test_slice_assign()
    test_slice_update()
