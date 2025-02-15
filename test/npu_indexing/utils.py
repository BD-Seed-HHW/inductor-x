import pdb

import torch
import torch_npu
import time

def create_profiler() :

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1, )

    prof = torch_npu.profiler.profile(
        activities=[
             torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=50, repeat=1, skip_first=10),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_dir"),
        experimental_config=experimental_config)
    prof.start()
    return prof

def benchmark_test(fn, fn_triton, args =(), name="gen_fn", times=10, repeat=10, profile=False):
    print(f"--------------------benchmark_{name} for {times * repeat} times--------------------")
    stream = torch.npu.current_stream()
    profiler = None
    if profile :
        profiler = create_profiler()

    stream.synchronize()
    if profile :
        profiler.start()
    start = time.perf_counter()
    for _ in range(times * repeat) :
        fn_triton(*args)
        if profile:
            profiler.step()
    stream.synchronize()
    end = time.perf_counter()
    if profile:
        profiler.stop()
    time_compiled = (end - start) / (times * repeat)
    time_compiled *= 1000000
    print(f"time_compiled:{time_compiled:.6f}")

    if profile :
        profiler = create_profiler()
    print(f"Runing eager {name} for {times * repeat} times")
    start = time.perf_counter()
    for _ in range(times * repeat) :
        fn(*args)
        if profile:
            profiler.step()
    stream.synchronize()
    end = time.perf_counter()
    time_eager = (end - start) / (times * repeat)
    time_eager *= 1000000
    print(f"time_eager:{time_eager:.6f}")
    accelerated = (time_eager - time_compiled)/time_compiled*100
    print(f"Accelerated: {accelerated:.4f}% eager takes {time_eager:.3f} us, triton takes {time_compiled:.3f} us")

    return time_eager, time_compiled