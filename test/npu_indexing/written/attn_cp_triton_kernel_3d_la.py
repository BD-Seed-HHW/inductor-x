import triton
import triton.language as tl
from triton.compiler import ASTSource, AttrsDescriptor
import torch
import torch_npu

NBLOCKS = 32
S_SUB_SIZE = 64


@triton.jit
def tl_fn_forward_update_la(prev_attn_out, prev_softmax_log_max_sum, cur_attn_out, cur_softmax_log_max_sum,
                            B: tl.constexpr, N: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
                            PREV_ATTN_NSTRIDE: tl.constexpr, PREV_SOFTMAX_NSTRIDE: tl.constexpr,
                            CUR_ATTN_NSTRIDE: tl.constexpr, CUR_SOFTMAX_NSTRIDE: tl.constexpr, S_NBLOCKS: tl.constexpr,
                            S_SUB: tl.constexpr):
    S_BLOCK: tl.constexpr = (S + S_NBLOCKS - 1) // S_NBLOCKS
    S_NSUB: tl.constexpr = (S_BLOCK + S_SUB - 1) // S_SUB
    LOOP_COUNT: tl.constexpr = (S_BLOCK * B * N + S_SUB - 1) // S_SUB
    block_idx = tl.program_id(0)

    s_block_start = block_idx * S_BLOCK
    SIMD_SIZE: tl.constexpr = S_SUB * D

    for loop_index in range(LOOP_COUNT):
        b = loop_index // (N * S_NSUB)
        n = (loop_index // S_NSUB) % N
        s_loop_start = (loop_index % S_NSUB) * S_SUB
        s = s_block_start + s_loop_start

        sd_offsets = D * s + tl.arange(0, SIMD_SIZE)
        s1_offsets = s + tl.arange(0, S_SUB)

        mask0 = None
        softmax_offsets = PREV_SOFTMAX_NSTRIDE * (b * N + n) + s1_offsets
        prev_softmax_local = tl.load(prev_softmax_log_max_sum + softmax_offsets, mask0)
        offsets = CUR_SOFTMAX_NSTRIDE * (b * N + n) + s1_offsets
        cur_softmax_local = tl.load(cur_softmax_log_max_sum + offsets, mask0)

        attn_offsets = PREV_ATTN_NSTRIDE * (b * N + n) + sd_offsets
        prev_attn_local = tl.load(prev_attn_out + attn_offsets, mask0)
        offsets = CUR_ATTN_NSTRIDE * (b * N + n) + sd_offsets
        cur_attn_local = tl.load(cur_attn_out + offsets, mask0)

        tmp0 = tl.exp(cur_softmax_local)
        tmp1 = tl.exp(prev_softmax_local)
        softmax_log_max_sum = tl.log(tmp0 + tmp1)
        tmp2 = (prev_softmax_local - softmax_log_max_sum).reshape(S_SUB, 1).broadcast_to(S_SUB, D)
        tmp3 = (cur_softmax_local - softmax_log_max_sum).reshape(S_SUB, 1).broadcast_to(S_SUB, D)

        attn_out = tl.exp(tmp2) * prev_attn_local.reshape(S_SUB, D) + (tl.exp(tmp3) * cur_attn_local.reshape(S_SUB, D))

        mask1 = None
        tl.store(prev_softmax_log_max_sum + softmax_offsets, softmax_log_max_sum, mask1)
        tl.store(prev_attn_out + attn_offsets, attn_out.reshape(SIMD_SIZE, ), mask1)


guards = {"dummy": None}


def forward_update_triton(prev_attn_out, prev_softmax_log_max_sum, cur_attn_out, cur_softmax_log_max_sum):
    (B, N, S, D) = cur_attn_out.shape
    PREV_ATTN_NSTRIDE = prev_attn_out.stride()[1]
    PREV_SOFTMAX_NSTRIDE = prev_softmax_log_max_sum.stride()[1]
    CUR_ATTN_NSTRIDE = cur_attn_out.stride()[1]
    CUR_SOFTMAX_NSTRIDE = cur_softmax_log_max_sum.stride()[1]

    compile_opt = ASTSource(
        fn=tl_fn_forward_update_la,
        signature={0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32'},
        constants={4: B, 5: N, 6: S, 7: D, 8: PREV_ATTN_NSTRIDE, 9: PREV_SOFTMAX_NSTRIDE, 10: CUR_ATTN_NSTRIDE,
                   11: CUR_SOFTMAX_NSTRIDE, 12: NBLOCKS, 13: S_SUB_SIZE},
        attrs=AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=()),
    )
    hash_code = "tt_attn_fw_update_la_{b}_{n}_{s}_{d}_{ns1}_{ns2}_{ns3}_{ns4}".format(b=B, n=N, s=S, d=D,
                                                                                      ns1=PREV_ATTN_NSTRIDE,
                                                                                      ns2=PREV_SOFTMAX_NSTRIDE,
                                                                                      ns3=CUR_ATTN_NSTRIDE,
                                                                                      ns4=CUR_SOFTMAX_NSTRIDE)

    compiled_func = guards.get(hash_code)
    if compiled_func == None:
        compiled_func = triton.compile(compile_opt, None, {"debug": True, "mix_mode": "aiv", "name": hash_code})
        guards[hash_code] = compiled_func

    device_id = cur_attn_out.device.index
    device = "npu:" + str(device_id)

    compiled_func[NBLOCKS, 1, 1](prev_attn_out, prev_softmax_log_max_sum, cur_attn_out, cur_softmax_log_max_sum)
