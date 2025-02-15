
import torch
from torch._inductor.codegen.common import register_backend_for_device, register_device_op_overrides
from . import config as npu_config
from torch_npu.utils._inductor import NPUDeviceOpOverrides

from torch_npu.utils._dynamo_device import NpuInterface
from torch_npu.npu.utils import device_count
from torch._dynamo.device_interface import register_interface_for_device, get_interface_for_device
from torch._inductor import lowering as inductor_lowering
from .lowering import _register_npu_inductor_fallbacks, make_reduction
from .decomposition import _register_npu_inductor_decompositons

print("perform inductor_npu patch")

def _inductor_register_backend_for_device():
    from .codegen.schduling import NPUTritonScheduling
    from .codegen.wrapper import NPUWrapperCodeGen
    register_backend_for_device('npu', NPUTritonScheduling, NPUWrapperCodeGen)

_inductor_register_backend_for_device()

## Override original inductor device overrides in torch_npu

# Not good implementation, but no other way
def get_current_raw_stream(device):
    return torch.npu.current_stream(device).npu_stream


class NewNPUDeviceOpOverrides(NPUDeviceOpOverrides):
    def import_get_raw_stream_as(self, name):
        return f"from inductor_npu import get_current_raw_stream as {name}"

def _inductor_register_device_op_overrides():
    register_device_op_overrides('npu', NewNPUDeviceOpOverrides())

_inductor_register_device_op_overrides()

## Override original dynamo device interface in torch_npu
class NewNpuInterface(NpuInterface):

    @staticmethod
    def is_available() -> bool:
        return device_count() > 0

    @staticmethod
    def get_compute_capability(device=None):
        # npu has no concept of cc. triton-npu compiler depends on subarch instead
        return torch.npu.get_device_name(device)

register_interface_for_device("npu", NewNpuInterface)
device = get_interface_for_device("npu")

from . import codegen

inductor_lowering.make_reduction = make_reduction
_register_npu_inductor_fallbacks()
_register_npu_inductor_decompositons()

#register fx_pass should be put behind of _register_npu_inductor_decompositons
#from .npu_indexing import fx_pass
from . import npu_fusion_attention_graph
from . import dynamo_patch3
