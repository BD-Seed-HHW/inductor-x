import triton
import triton.language as tl
import torch
import torch_npu


@triton.jit
def load_store_kernel(in_out_ptr0, in_out_ptr1, in_ptr0):
    XBLOCK = 32
    for x0 in range(0, XBLOCK):
        tmp0 = tl.load(in_ptr0 + x0 + tl.arange(0, 1), None)
        tmp1 = x0 * 2.0
        tl.store(in_out_ptr0 + x0 + tl.arange(0, 1), tmp0, None)
        tl.store(in_out_ptr1 + x0 + tl.arange(0, 1), tmp1, None)


def load_store(in0, stream):
    out0 = torch.empty_strided((32,), (1,), device="npu", dtype=torch.float32)
    out1 = torch.empty_strided((32,), (1,), device="npu", dtype=torch.float32)
    kernel = load_store_kernel.warmup(out0, out1, in0, grid=(1,))
    kernel._init_handles()
    kernel[(1, 1, 1)](out0, out1, in0, stream=stream)
    return out0, out1


device = torch.npu.current_device()
stream = torch.npu.current_stream(device).npu_stream
in0 = torch.arange(0, 32, device="npu") + 100
out0, out1 = load_store(in0, stream)
print(out0, out1)
