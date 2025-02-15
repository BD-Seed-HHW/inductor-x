# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import torch.nn.functional as F
import torch
import torch_npu
import inductor_npu


inductor_npu.config.enable_npu_indexing=True
from test2.npu_indexing.utils import benchmark_test
def test_sum(view_12, embedding_1, slice_11):
    # 原网络
        
    permute_7 = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
    unsqueeze_4 = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None

    add_5 = torch.ops.aten.add.Tensor(unsqueeze_4, slice_11);  slice_8 = slice_11 = None
    add_6 = torch.ops.aten.add.Tensor(view_12, add_5);  view_12 = None
    return add_6

device='npu'

if __name__ == "__main__":
    embedding_1 = torch.randn((512, 512, 64), device=device, dtype=torch.float32)
    primals_221 = torch.randn((1, 1, 1, 512), device=device, dtype=torch.float32)
    view_12 = torch.randn((1, 64, 512, 512), device=device, dtype=torch.float32)
    slice_11 = torch.randn((1, 1, 1, 512), device=device, dtype=torch.float32)
       
    ref = test_sum(view_12, embedding_1, primals_221)
    func = torch.compile(test_sum, backend="inductor", dynamic=False)
    calc = func(view_12, embedding_1, primals_221)

    torch.testing.assert_close(ref, calc, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(ref, calc, rtol=1e-3, atol=1e-3)

    print("valid ok")
    benchmark_test(test_sum, func, args=(view_12, embedding_1, primals_221),
                   name="issue57", times=10, repeat=10, profile=False)


