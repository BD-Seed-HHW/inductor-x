import os  # noqa: C101
import sys
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING
from triton.runtime.driver import driver
from torch._inductor import config
enable_npu_indexing = True

config.triton.unique_kernel_names = True
# avoid test_opensora_cases_model_16_forward  reinterpre_tensor issue
config.allow_buffer_reuse = False
#inductor debug switch
config.trace.enabled = True

# npu hardware params from trion
target = driver.active.get_current_target()
device = driver.active.get_current_device()
prop = driver.active.utils.get_device_properties(device)

num_cube_core = prop["num_aicore"]
num_vector_core = prop["num_aicore"]

# unit byte
npu_block = 32



if ("Ascend910B" in target.arch):
    num_vector_core = num_cube_core * 2


