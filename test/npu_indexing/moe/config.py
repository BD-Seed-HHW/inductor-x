import torch


class Config:
    def __init__(self,
                 hidden_size,
                 num_experts=1,
                 ep_size=1,
                 topk=1,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 aux_loss_coef=0.0,
                 z_loss_coef=0.0,
                 noisy_gate_policy=None,
                 use_rts=True,
                 enable_token_rearrange_opt=False,
                 gating_optimized=False,
                 reshape_index_select=None,
                 ):
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.topk = topk
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.aux_loss_coef = aux_loss_coef
        self.z_loss_coef = z_loss_coef
        self.noisy_gate_policy = noisy_gate_policy
        self.enable_token_rearrange_opt = enable_token_rearrange_opt
        self.use_rts = use_rts
        self.gating_optimized = gating_optimized
        self.reshape_index_select = None
        if reshape_index_select:
            self.reshape_index_select = torch.tensor(reshape_index_select, dtype=torch.int32,
                                                     device=torch.npu.current_device())