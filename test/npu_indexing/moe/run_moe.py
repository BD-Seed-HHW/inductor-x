import torch
from config import Config
from typing import Any
from typing import Callable, Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from utils import gumbel_rsample, get_capacity, old_get_capacity, einsum, _one_hot_to_float, MoEAuxLossAutoScaler, _capacity


def apply_aux_loss(config, gates, mask1):
    num_experts = int(gates.shape[1])
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts
    if config.aux_loss_coef > 0:
        l_aux = l_aux * config.aux_loss_coef
        gates = MoEAuxLossAutoScaler.apply(gates, l_aux)
    return gates, l_aux


def top2gating(logits: Tensor, config: Config) -> Tuple[Tensor, ...]:
    """Implements Top2Gating on logits."""
    # apply z loss
    # logits = apply_z_loss(config, logits)   #确认配置为0，暂时未生效

    # everything is in fp32 in this function
    token_sel_expert_weights = F.softmax(logits, dim=1)

    if config.reshape_index_select is not None:
        token_sel_expert_weights = token_sel_expert_weights[:, config.reshape_index_select]

    num_experts = int(token_sel_expert_weights.shape[1])

    if config.gating_optimized:
        capacity, capacity_host = get_capacity(token_sel_expert_weights, config.capacity_factor * 2,
                                               config.min_capacity)
    else:
       num_tokens = token_sel_expert_weights.shape[0]
       num_experts = token_sel_expert_weights.shape[1]
       capacity = _capacity(num_tokens, num_experts, 
                                    config.capacity_factor * 2, 
                                    config.min_capacity)
       # capacity = old_get_capacity(token_sel_expert_weights,
       #                             torch.tensor(config.capacity_factor * 2),
       #                             torch.tensor(config.min_capacity))

    _, selected_experts = torch.topk(token_sel_expert_weights, config.topk, dim=-1)
    mask = F.one_hot(selected_experts, num_classes=num_experts)
    first_expert_mask = mask[:, 0, :]
    second_expert_mask = mask[:, 1, :]

    # Compute locations in capacity buffer
    locations_in_first_expert = torch.cumsum(first_expert_mask, dim=0) - 1
    locations_in_second_expert = torch.cumsum(second_expert_mask, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations_in_second_expert += torch.sum(first_expert_mask, dim=0, keepdim=True)

    # Compute l_aux
    token_sel_expert_weights, l_aux = apply_aux_loss(config, token_sel_expert_weights, first_expert_mask)

    # Remove locations outside capacity from mask
    capacity_tensor = torch.tensor(capacity, dtype=torch.int32)
    first_expert_mask *= torch.lt(locations_in_first_expert, capacity_tensor)
    second_expert_mask *= torch.lt(locations_in_second_expert, capacity_tensor)

    # Store the capacity location for each token
    token_idx_in_first_expert = torch.sum(locations_in_first_expert * first_expert_mask, dim=1)
    token_idx_in_second_expert = torch.sum(locations_in_second_expert * second_expert_mask, dim=1)

    # Normalize gate probabilities
    first_expert_mask_float = first_expert_mask.float()
    second_expert_mask_float = second_expert_mask.float()
    token_first_exp_weights, token_first_exp_idx = torch.max(token_sel_expert_weights * first_expert_mask_float,
                                                             dim=1)
    token_second_exp_weights, token_second_exp_idx = torch.max(token_sel_expert_weights * second_expert_mask_float,
                                                               dim=1)
    denom_s = token_first_exp_weights + token_second_exp_weights
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    token_first_exp_weights /= denom_s
    token_second_exp_weights /= denom_s

    if config.enable_token_rearrange_opt:
        token_rearranged_first_ec_idx = token_first_exp_idx.int() * capacity + token_idx_in_first_expert.int()
        token_rearranged_second_ec_idx = token_second_exp_idx.int() * capacity + token_idx_in_second_expert.int()

        token_sel_first_exp_int_mask = first_expert_mask * 2
        token_sel_second_exp_int_mask = second_expert_mask

        if config.gating_optimized:
            expert_sel_top_c_token_idx = torch.topk(token_sel_first_exp_int_mask + token_sel_second_exp_int_mask,
                                                    k=capacity_host,
                                                    dim=0,
                                                    sorted=True)[1]
            expert_sel_token_idx = expert_sel_top_c_token_idx.t().reshape(num_experts * capacity_host)
        else:
           # capacity_value = capacity.item()
            expert_sel_top_c_token_idx = torch.topk(token_sel_first_exp_int_mask + token_sel_second_exp_int_mask,
                                                    k = capacity, 
                                                    #k=capacity.item(),
                                                    dim=0,
                                                    sorted=True)[1]
            expert_sel_token_idx = expert_sel_top_c_token_idx.t()
            #expert_sel_token_idx = expert_sel_token_idx.reshape(num_experts * capacity)

        token_rearranged_ec_idx = torch.cat([token_rearranged_first_ec_idx, token_rearranged_second_ec_idx], dim=0)
        token_exp_weights = torch.cat([token_first_exp_weights, token_second_exp_weights], dim=0)

        return l_aux, token_rearranged_ec_idx, token_exp_weights, expert_sel_token_idx
    else:

        # Calculate combine_weights and dispatch_mask
        gates1 = einsum("s,se->se", token_first_exp_weights, first_expert_mask_float)
        gates2 = einsum("s,se->se", token_second_exp_weights, second_expert_mask_float)
        locations1_sc = _one_hot_to_float(token_idx_in_first_expert, capacity)
        locations2_sc = _one_hot_to_float(token_idx_in_second_expert, capacity)
        combine1_sec = einsum("se,sc->sec", gates1, locations1_sc)
        combine2_sec = einsum("se,sc->sec", gates2, locations2_sc)
        combine_weights = combine1_sec + combine2_sec
        dispatch_mask = combine_weights.bool()

        return l_aux, combine_weights, dispatch_mask