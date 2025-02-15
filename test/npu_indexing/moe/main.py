from config import Config
from run_moe import top2gating 
import torch
import time
import logging

import torch_npu
import sys
sys.path.append("../../..")
import inductor_npu
inductor_npu.config.enable_npu_indexing = True
from inductor_npu import npu_indexing


conf = Config(capacity_factor = 1.1, gating_optimized=False, topk=2, hidden_size=12288, enable_token_rearrange_opt=True)
#torch.cuda.set_device(3)


"""
#==============================add for profiling============================
experimental_config = torch_npu.profiler._ExperimentalConfig(
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
    )
prof = torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=28),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("/home/h00815022/result_dir"),
    experimental_config=experimental_config)
prof.start()
"""
device = "npu"
random_tensor = torch.randn((4096, 8), requires_grad=True, device = device, dtype=torch.float32)
#random_tensor = torch.randn(4, 4)
#random_tensor = random_tensor.cuda()
model = torch.compile(top2gating, backend="inductor", dynamic=False)
torch._dynamo.disallow_in_graph(torch.nn.functional.one_hot)
for i in range(100):
 
    total_loss = []
    l_aux, token_rearranged_ec_idx, token_exp_weights, expert_sel_token_idx = model(random_tensor, conf)
    print(f"loop{i} l_aux:{l_aux} ec_index:{token_rearranged_ec_idx.shape} exp_weights:{token_exp_weights.shape} "
          f"token_idx:{expert_sel_token_idx.shape}")
    output = token_exp_weights
    output.backward(torch.ones_like(output))
         
    #optimizer.step(
    
    print(f"finished to loop{i}")
    time.sleep(1)