
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

import torch_npu
import inductor_npu
import logging
#torch._logging.set_logs(inductor=logging.DEBUG)
from torch.nn import CrossEntropyLoss

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()

def test_fx_graph(view_12, embedding_1):
    # 原网络
    permute_7 = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);
    unsqueeze_2 = torch.ops.aten.unsqueeze.default(permute_7, 0);
    slice_3 = torch.ops.aten.slice.Tensor(unsqueeze_2, 0, 0, 9223372036854775807);
    slice_4 = torch.ops.aten.slice.Tensor(slice_3, 1, 0, 9223372036854775807);
    slice_5 = torch.ops.aten.slice.Tensor(slice_4, 2, -128, 9223372036854775807);
    slice_6 = torch.ops.aten.slice.Tensor(slice_5, 3, 0, 9223372036854775807);
    add_5 = torch.ops.aten.add.Tensor(view_12, slice_6);
    view_13 = torch.ops.aten.view.default(add_5, [384, 128, 128]);
    view_14 = torch.ops.aten.view.default(view_13, [64, 6, 128, 128]);
    return view_14

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import split_scan_grid, start_graph, end_graph
from inductor_npu.npu_triton_heuristics import grid
from inductor_npu import get_current_raw_stream as get_raw_stream


async_compile.wait(globals())
del async_compile


buf85 = empty_strided((64, 6, 128, 128), (98304, 16384, 128, 1), device='npu', dtype=torch.float32)
buf84 = empty_strided((128, 128, 6), (768, 6, 1), device='npu', dtype=torch.float32) # embedding output
# Source Nodes: [add_5, add_6], Original ATen: [aten.add]
stream0 = get_raw_stream(0)
# triton_unk_fused_add_0[32, 1, 1](arg2_1, arg0_1, arg1_1, buf0, 64, 512, 512, 64, 32, 256)
# triton_unk_fused_add_10.run(buf85, buf84, 64, 6, 16384, grid=grid(64, 6, 16384), stream=stream0)
model = torch.compile(test_fx_graph, backend="inductor")
data_t = model(buf85, buf84)
data = test_fx_graph(buf85, buf84)
assert torch.allclose(data, data_t, atol=1e-3, rtol=1e-3), "Tensors are not close enough!"

print("data validation passed")

