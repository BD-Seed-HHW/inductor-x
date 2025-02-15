# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import torch
import torch_npu
import sys
sys.path.append("../..")
import inductor_npu


def argmax(a,  dim):
    return torch.argmax(a, dim)

torch_npu.npu.utils.set_device(1)
device = "npu"
def test_argmax(shape, dim) :
    print(f"start to test argmax on shape:{shape} dim:{dim} ")
    a = torch.randn(shape, requires_grad=False, dtype=torch.float32, device=device)

    argmax_triton = torch.compile(argmax, backend="inductor")
    r = argmax(a, dim)
    r1 = argmax_triton(a, dim)
    torch.testing.assert_close(r, r1, rtol=1e-3, atol=1e-3)
    print("argmax validation passed")
#test_argmax((16, 512, 64), -1)
test_argmax((512, 64), -1)
