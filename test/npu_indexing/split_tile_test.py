
import triton
import triton.language as tl
import torch
import torch_npu

@triton.jit
def triton_unk_fused_clamp_9(in_ptr0, out_ptr0, x0_numel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    x0_offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1 = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop1 in range(loops1):
        x0_prime = x0_offset + (loop1 * XBLOCK_SUB) + base1
        x0 = x0_offset + (loop1 * XBLOCK_SUB) + base1
        x0_mask = x0 < x0_numel
        x0_prime_mask = x0_prime < x0_numel
        tmp0 = tl.load(in_ptr0 + (x0), x0_mask)
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tl.maximum(tmp0, tmp1)
        tmp3 = tl.full([1], 7, tl.int64)
        tmp4 = tl.minimum(tmp2, tmp3)
        tl.store(out_ptr0 + (x0), tmp4, x0_mask)



if __name__ == '__main__':
    primals_46 = torch.empty_strided((13,), (1,), device='npu:1', dtype=torch.int64)
    buf103 = torch.empty_strided((13,), (1,), device='npu', dtype=torch.int64)
    # Source Nodes: [start_positions], Original ATen: [aten.clamp]
    kernel = triton_unk_fused_clamp_9.warmup(primals_46, buf103, 13, XBLOCK=8, XBLOCK_SUB=8, grid=(2,1,1))
    kernel._init_handles()



