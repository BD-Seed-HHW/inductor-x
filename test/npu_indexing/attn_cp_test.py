# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.



import torch
import torch_npu
import sys
sys.path.append("../..")
import inductor_npu

from einops import rearrange
from test2.npu_indexing.utils import benchmark_test
F32_BLK_SIZE = 8

def trans_BNSD2SBH(x):
    #(S, B, H, N) = (4096, 1, 96, 12)
    """Trans data layout from BNSD to SBH"""
    return rearrange(x, 'b n s d -> s b (n d)').contiguous()
   # x = x.permute(2,0,1,3)
   # return x.reshape(S, B, H)

def broadcast_and_trans_BNSD2SBH(x, h):

    """broadcast and trans a tensor from [b, n, s, 8] to [s, b, h]"""
    n = x.shape[1]
    d = h // n
    # [b, n, s, 8] -> [b, n, s, d]
    new_x = x[..., 0].unsqueeze(3)
    new_x = new_x.repeat(1, 1, 1, d)
    #new_x = x.repeat(1,1,1, d // F32_BLK_SIZE)
    return trans_BNSD2SBH(new_x)



def forward_update(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                   cur_attn_out, cur_softmax_max, cur_softmax_sum):
    # update softmax_max
    #print("softmax_max shape:", prev_softmax_max.shape, "softmax_max dtype:", prev_softmax_max.dtype )
    #print("prev_attn_out shape:", prev_attn_out.shape, "prev_attn_out dtype:", prev_attn_out.dtype )

    org_dtype = prev_attn_out.dtype
    softmax_max = torch.maximum(prev_softmax_max, cur_softmax_max)
    prev_scale = torch.exp(prev_softmax_max - softmax_max)
    cur_scale = torch.exp(cur_softmax_max - softmax_max)
    # update softmax_sum
    prev_softmax_sum_scaled = prev_softmax_sum * prev_scale
    cur_softmax_sum_scaled = cur_softmax_sum * cur_scale
    softmax_sum = prev_softmax_sum_scaled + cur_softmax_sum_scaled
    # out updating scale
    prev_out_scale = prev_softmax_sum_scaled / softmax_sum
    cur_out_scale = cur_softmax_sum_scaled / softmax_sum
    # [b, n, s, 8] -> [s, b, h]
    prev_out_scale_sbh = broadcast_and_trans_BNSD2SBH(prev_out_scale, prev_attn_out.shape[-1])
    cur_out_scale_sbh = broadcast_and_trans_BNSD2SBH(cur_out_scale, prev_attn_out.shape[-1])
    # update output
    attn_out = prev_attn_out * prev_out_scale_sbh + cur_attn_out * cur_out_scale_sbh
    attn_out = attn_out.to(org_dtype)
    return attn_out, softmax_max, softmax_sum


def data_validation(forward_update_triton, prev_softmax_max, cur_softmax_max, prev_softmax_sum, cur_softmax_sum , prev_attn_out, cur_attn_out):

    (attn_out, softmax_max, softmax_sum) = forward_update(prev_attn_out, prev_softmax_max, prev_softmax_sum,
                                                          cur_attn_out,
                                                          cur_softmax_max, cur_softmax_sum)

    (tt_attn_out, tt_softmax_max, tt_softmax_sum) = forward_update_triton(prev_attn_out,
            prev_softmax_max, prev_softmax_sum, cur_attn_out, cur_softmax_max, cur_softmax_sum)

    try :
        torch.testing.assert_close(softmax_max, tt_softmax_max)
        print("max comparition passed.")
        torch.testing.assert_close(softmax_sum, tt_softmax_sum)
        print("sum comparition passed.")
        torch.testing.assert_close(attn_out, tt_attn_out )
        print("atten comparition passed.")

    except Exception as e :
        print(e)
        print("comparison not passed")
    print(f"proving finished, attn shape:{prev_attn_out.shape}, stride:{prev_attn_out.stride(),cur_attn_out.stride()}, softmax shape:{prev_softmax_sum.shape}, stride:{prev_softmax_sum.stride(), cur_softmax_sum.stride()}")


if __name__ == "__main__":
    inductor_npu.config.enable_npu_indexing = True
    torch_npu.npu.utils.set_device(2)
    #(S, B, H, N) = (4096, 1, 192, 12)
    (S, B, H, N) = (4096,1,1536,12)
    DS = 2 * S
    DTYPE_ATTN = torch.float32
    DTYPE = torch.float32


    prev_attn_out = torch.randn((DS, B, H), dtype=DTYPE_ATTN).npu()
    prev_softmax_max = torch.rand((B, N, DS), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)
    prev_softmax_sum = torch.rand((B, N, DS), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)
    cur_attn_out = torch.randn((DS, B, H), dtype=DTYPE_ATTN).npu()
    cur_softmax_max = torch.rand((B, N, DS), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)
    cur_softmax_sum = torch.rand((B, N, DS), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)
    forward_update_triton_2s = torch.compile(forward_update, backend="inductor", options={"aggressive_fusion" :True })
    print("--------------------prove_forward_update:2S----------------------")
    data_validation(forward_update_triton_2s, prev_softmax_max, cur_softmax_max, prev_softmax_sum, cur_softmax_sum,
                    prev_attn_out,
                    cur_attn_out)

    prev_attn_out_s = prev_attn_out.view(2, S, B, H)[1]
    prev_softmax_max_s = prev_softmax_max.view(B, N, 2, S, F32_BLK_SIZE)[:, :, 1, :, :]
    prev_softmax_sum_s = prev_softmax_sum.view(B, N, 2, S, F32_BLK_SIZE)[:, :, 1, :, :]
    cur_attn_out_s = torch.randn((S, B, H), dtype=DTYPE_ATTN).npu()
    cur_softmax_max_s = torch.rand((B, N, S), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)
    cur_softmax_sum_s = torch.rand((B, N, S), dtype=DTYPE).npu().unsqueeze(3).repeat(1, 1, 1, F32_BLK_SIZE)
    # print("--------------------prove_forward_update:1S---------------------- ")
    # forward_update_triton = torch.compile(forward_update, backend="inductor")
    # data_validation(forward_update_triton, prev_softmax_max_s, cur_softmax_max_s, prev_softmax_sum_s,
    #                 cur_softmax_sum_s, prev_attn_out_s, cur_attn_out_s)


    benchmark_test(forward_update, forward_update_triton_2s, args=(prev_attn_out,
                                                                prev_softmax_max, prev_softmax_sum, cur_attn_out,
                                                                cur_softmax_max, cur_softmax_sum),
                   name="forward_update_2s", times=10, repeat=10, profile = False)

    # benchmark_test(forward_update, forward_update_triton, args=(prev_attn_out_s,
    #                                                             prev_softmax_max_s, prev_softmax_sum_s, cur_attn_out_s,
    #                                                             cur_softmax_max_s, cur_softmax_sum_s),
    #                name="forward_update_s", times=10, repeat=100, profile = False)





