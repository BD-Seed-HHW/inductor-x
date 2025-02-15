#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.


print("perform npu_indexing patch ")

from torch._inductor.ir import Reduction,LoopBody
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor import sizevars
from torch._inductor.codegen.triton import TritonKernel

from inductor_npu.codegen._sizevars import simplify
from inductor_npu.codegen.ir import (num_splits,loopbody__call__,transform_dims_in_indexing,
                                     substituted_dims_in_indexing)
from inductor_npu.codegen.triton import is_compatible
from inductor_npu.codegen.triton import group_fn, select_index_dtype, select_tiling
#from ..npu_indexing.graph import run_node
#graph
#GraphLowering.run_node = run_node
#common
#ir
Reduction.num_splits = num_splits
setattr(LoopBody, 'transform_dims_in_indexing', transform_dims_in_indexing)
setattr(LoopBody, 'substituted_dims_in_indexing', substituted_dims_in_indexing)

LoopBody.__call__=loopbody__call__
#need to enable this to speedup attn_cp_test
#ComputedBuffer.simplify_and_reorder = simplify_and_reorder
#triton scheduling
TritonScheduling.group_fn = group_fn
TritonScheduling.select_index_dtype = select_index_dtype
TritonScheduling.select_tiling = select_tiling
#triton kernel
setattr(TritonKernel, 'is_compatible', is_compatible )

#util
sizevars.SizeVarAllocator.simplify = simplify