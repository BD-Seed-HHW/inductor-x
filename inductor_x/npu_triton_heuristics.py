# This file is based on triton_heuristics with heuristics designed for NPU
import logging
import os
import functools
import copy
import pdb
from typing import Any, Callable, List, Optional

import torch
import re

from torch._inductor import config
import hashlib
from .codegen.tile_generator import TileGenerator


from torch._inductor.triton_heuristics import (
    CachingAutotuner,
    HeuristicType,
    unique_configs,
    hash_configs,
    load_cached_autotuning,
    Config,
    ASTSource,
    _find_names,
    get_first_attr,
    collected_calls,
    json,

)

from torch._inductor.utils import (
    create_bandwidth_info_str,
    get_num_bytes,

)
from .codegen.split_tiling import SplitTiling
import triton

try:
    from triton.backends.compiler import GPUTarget
    from triton.runtime.autotuner import OutOfResources
    import torch.autograd.profiler as autograd_profiler
except ImportError:
    GPUTarget = None
    OutOfResources = None
    autograd_profiler = None

import inductor_npu

log = logging.getLogger(__name__)
log.level = logging.DEBUG
class NPUCachingAutotuner(CachingAutotuner):
    def __init__(
        self,
        fn,
        triton_meta,  # passed directly to triton
        configs,
        save_cache_hook,
        mutated_arg_names,
        heuristic_type,
        size_hints=None,
        inductor_meta=None,  # metadata not relevant to triton
        custom_kernel=False,  # whether the kernel is inductor-generated or custom
    ):
        super().__init__(fn, triton_meta, configs, save_cache_hook, mutated_arg_names, heuristic_type, size_hints, inductor_meta, custom_kernel)
        self.gpu_device.get_raw_stream=inductor_npu.get_current_raw_stream


    # don't print exceptions when UB exception thrown by underlying compiler
    def precompile(self, warm_cache_only_with_cc=None):
        with self.lock:
            if self.launchers:
                return
            self.launchers = []
            compiled_binaries = []
            save_configs = []
            latest_config = None
            for c in self.configs:
                try:
                    latest_config = c
                    compiled_binary, launcher = self._precompile_config(
                        c, warm_cache_only_with_cc
                    )
                    if (compiled_binary is None):
                        continue
                except OutOfResources:
                    # Skip the config if we run out of resource
                    continue
                self.launchers.append(launcher)
                compiled_binaries.append(compiled_binary)
                if (compiled_binary is not None):
                    save_configs.append(c)
                    break
            # remove compile failure tiling case
            self.configs = save_configs
            if len(self.launchers) == 0:
                log.exception(
                    "Triton compilation failed: %s\n%s\nmetadata: %s",
                    self.inductor_meta.get("kernel_name", "triton_"),
                    self.fn.src,
                    latest_config.kwargs,
                )
                raise RuntimeError(
                    "No valid triton configs. Report a fatal compilation error"
                )
            self.configs = None

    # to add the line  options["mix_mode"] = "aiv"
    # to filter out some options on cfg used for gird, but not for constants
    def _precompile_config(self, cfg: Config, warm_cache_only_with_cc: Optional[int]):
        """Ahead of time compile a given autotuner config."""
        compile_meta = copy.deepcopy(self.triton_meta)
        for k, v in cfg.kwargs.items():
            if k not in self.fn.arg_names :
                continue
            index = self.fn.arg_names.index(k)
            compile_meta["constants"][self.fn.arg_names[index]] = v
        # for higher version triton
        kwargs_list = [k for k, v in cfg.kwargs.items()]
        for i, arg in enumerate(self.fn.arg_names):
            if arg in kwargs_list:
                continue
            name = self.fn.arg_names[i]
            value = compile_meta["signature"][i]
            del compile_meta["signature"][i]
            compile_meta["signature"][name] = value

        compile_meta["num_warps"] = cfg.num_warps
        compile_meta["num_stages"] = cfg.num_stages
        compile_meta["debug"] = (
                config.assert_indirect_indexing and torch.version.hip is None
        )

        # Setting device_type="hip" required on ROCm to pass down to triton
        compile_meta["device_type"] = (
            self.device_type if torch.version.hip is None else "hip"
        )
        if warm_cache_only_with_cc:
            cc = warm_cache_only_with_cc
        else:
            # Use device_type 'cuda' for both cuda and hip devices to retrieve
            # the compute capability.
            device_type = self.device_type if torch.version.hip is None else "cuda"
            device_id = compile_meta["device"]
            device = torch.device(device_type, device_id)
            cc = self.gpu_device.get_compute_capability(device)

        compile_meta["cc"] = cc

        if ASTSource:
            compile_args = (
                ASTSource(
                    self.fn,
                    compile_meta["signature"],
                    compile_meta["constants"],
                    None,
                    #compile_meta["configs"][0],
                ),
            )

            if GPUTarget:
                target = GPUTarget(compile_meta["device_type"], cc, 0)
            else:
                target = (compile_meta["device_type"], cc)

            options = {
                "num_warps": compile_meta["num_warps"],
                "num_stages": compile_meta["num_stages"],
                "debug": compile_meta["debug"],
            }
            # Note: currently force to generate vector kernels only
            if self.device_type == "npu":
                options["mix_mode"] = "aiv"
            compile_kwargs = {
                "target": target,
                "options": options,
            }
        else:
            compile_args = (self.fn,)
            compile_kwargs = compile_meta

        if warm_cache_only_with_cc:
            return (
                triton.compile(*compile_args, **compile_kwargs),
                None,
            )

        # load binary to the correct device
        with self.gpu_device.device(compile_meta["device"]):  # type: ignore[attr-defined]
            # need to initialize context
            self.gpu_device.synchronize(self.gpu_device.current_device())

            try:
                binary = triton.compile(*compile_args, **compile_kwargs)
            except Exception:
                # compile failed don't need raise error for npu
                return None, None
            binary._init_handles()

        call_args = [
            arg
            for i, arg in enumerate(self.fn.arg_names)
            if i not in self.fn.constexprs
        ]
        def_args = [name for name in self.fn.arg_names if name not in cfg.kwargs]
        scope = {
            "grid_meta": cfg.kwargs,
            "bin": binary,
            "launch_enter_hook": binary.launch_enter_hook,
            "launch_exit_hook": binary.launch_exit_hook,
            "metadata": binary.metadata,
            "torch": torch,
            "set_device": self.gpu_device.set_device,
            "current_device": self.gpu_device.current_device,
        }

        scope["runner"] = get_first_attr(binary, "run", "c_wrapper")
        scope["function"] = get_first_attr(binary, "function", "cu_function")
        scope["cta_args"] = (
            (binary.num_ctas, *get_first_attr(binary, "cluster_dims", "clusterDims"))
            if hasattr(binary, "num_ctas")
            else (
                (binary.metadata.num_ctas, *binary.metadata.cluster_dims)
                if hasattr(binary, "metadata")
                else ()
            )
        )
        scope["num_warps"] = (
            binary.num_warps
            if hasattr(binary, "num_warps")
            else binary.metadata.num_warps
        )
        binary_shared = (
            binary.shared if hasattr(binary, "shared") else binary.metadata.shared
        )
        scope["shared"] = binary_shared

        exec(
            f"""
                def launcher({', '.join(def_args)}, grid, stream):
                    if callable(grid):
                        grid_0, grid_1, grid_2 = grid(grid_meta)
                    else:
                        grid_0, grid_1, grid_2 = grid

                    bin[grid_0, grid_1, grid_2](
                                {', '.join(call_args)},
                                stream=stream)
                    return bin
                """.lstrip(),
            scope,
        )

        launcher = scope["launcher"]
        launcher.config = cfg
        launcher.n_regs = getattr(binary, "n_regs", None)
        launcher.n_spills = getattr(binary, "n_spills", None)
        launcher.shared = binary_shared
        launcher.store_cubin = config.triton.store_cubin
        # store this global variable to avoid the high overhead of reading it when calling run
        if launcher.store_cubin:
            launcher.fn = self.fn
            launcher.bin = binary

        return binary, launcher

class NPUDebugAutotuner(NPUCachingAutotuner):
    def __init__(self, *args, regex_filter="", **kwargs):
        self.regex_filter = regex_filter
        super().__init__(*args, **kwargs)
        self.cached = None

    def run(self, *args, grid, stream):
        possible_names = _find_names(self)
        kernel_name = f"{max(possible_names, key=len)}"
        if not re.match(self.regex_filter, kernel_name):
            return
        super().run(*args, grid=grid, stream=stream)
        (launcher,) = self.launchers

        if self.cached is None:
            ms = self.bench(launcher, *args, grid=grid)
            num_in_out_ptrs = len(
                [
                    arg_name
                    for arg_name in self.fn.arg_names
                    if arg_name.startswith("in_out_ptr")
                ]
            )
            num_gb = get_num_bytes(*args, num_in_out_args=num_in_out_ptrs) / 1e9
            gb_per_s = num_gb / (ms / 1e3)
            self.cached = (ms, num_gb, gb_per_s, kernel_name)
        else:
            ms, num_gb, gb_per_s, kernel_name = self.cached
        collected_calls.append((ms, num_gb, gb_per_s, kernel_name))
        print(
            create_bandwidth_info_str(ms, num_gb, gb_per_s, suffix=f" \t {kernel_name}")
        )


def cached_autotune(
    size_hints: Optional[List[int]],
    configs: List[Config],
    triton_meta,
    heuristic_type,
    filename=None,
    inductor_meta=None,
    custom_kernel=False,
):
    """
    A copy of triton.autotune that calls our subclass.  Our subclass
    has additional debugging, error handling, and on-disk caching.
    """
    configs = unique_configs(configs)
    assert len(configs) == 1 or filename
    save_cache_hook: Optional[Callable[[Any, Any], Any]]
    inductor_meta = {} if inductor_meta is None else inductor_meta

    # on disk caching logic and/or remote caching
    if filename is not None and (len(configs) > 1 or config.coordinate_descent_tuning):
        configs_hash = hash_configs(configs)

        cache_filename = None
        remote_cache = None
        remote_cache_key = None
        if config.use_autotune_local_cache:
            cache_filename = os.path.splitext(filename)[0] + ".best_config"
        if config.use_autotune_remote_cache or (
            config.is_fbcode()
            and torch._utils_internal.justknobs_check(
                "pytorch/autotune_remote_cache:enable"
            )
        ):
            backend_hash = inductor_meta.get("backend_hash", None)
            if backend_hash is not None:
                key = backend_hash + configs_hash + "autotune-best-config"
                key = hashlib.sha256(key.encode("utf-8")).hexdigest()

                try:
                    if config.is_fbcode():
                        remote_cache = (
                            triton.runtime.fb_memcache.FbMemcacheRemoteCacheBackend(
                                key, is_autotune=True
                            )
                        )
                    else:
                        remote_cache = triton.runtime.cache.RedisRemoteCacheBackend(key)
                except Exception:
                    remote_cache = None
                    log.warning("Unable to create a remote cache", exc_info=True)
                # we already sha256 hash the source contents
                remote_cache_key = os.path.basename(filename)
            else:
                log.debug(
                    "backend_hash is not passed on the inductor_meta, unable to use autotune remote cache"
                )

        best_config = None
        if cache_filename is not None and os.path.exists(cache_filename):
            with open(cache_filename) as fd:
                best_config = json.loads(fd.read())
        elif remote_cache is not None and remote_cache_key is not None:
            cache_outs = remote_cache.get([remote_cache_key])
            cache_out = cache_outs.get(remote_cache_key, None)
            best_config = json.loads(cache_out) if cache_out else None

        best_config = load_cached_autotuning(best_config, configs_hash, configs)
        if best_config:
            configs = [best_config]

        def save_cache_hook(cfg, found_by_coordesc=False):
            data = json.dumps(
                {
                    **cfg.kwargs,
                    "num_warps": cfg.num_warps,
                    "num_stages": cfg.num_stages,
                    "configs_hash": configs_hash,
                    "found_by_coordesc": found_by_coordesc,
                }
            )
            if cache_filename is not None:
                with open(cache_filename, "w") as fd:
                    fd.write(data)
            if remote_cache is not None and remote_cache_key is not None:
                remote_cache.put(remote_cache_key, data)

            if log.isEnabledFor(logging.DEBUG):
                type_str = "coordesc" if found_by_coordesc else "heuristic"
                log.debug("Save %s tuning result to %s", type_str, cache_filename)

    else:
        save_cache_hook = None

    mutated_arg_names = inductor_meta.pop("mutated_arg_names", ())

    def decorator(fn):
        # Remove XBLOCK from config if it's not a function argument.
        # This way, coordinate descent tuning will not try to tune it.
        #
        # Context: When TritonKernel.no_x_dim is True, we hardcode XBLOCK to 1.
        import inspect

        if "XBLOCK" not in inspect.signature(fn.fn).parameters:
            for tconfig in configs:
                if "XBLOCK" in tconfig.kwargs:
                    assert tconfig.kwargs["XBLOCK"] == 1
                    tconfig.kwargs.pop("XBLOCK")

        if config.profile_bandwidth:
            return NPUDebugAutotuner(
                fn,
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                regex_filter=config.profile_bandwidth_regex,
                configs=configs,
                save_cache_hook=save_cache_hook,
                mutated_arg_names=mutated_arg_names,
                heuristic_type=heuristic_type,
                size_hints=size_hints,
                custom_kernel=custom_kernel,
            )
        return NPUCachingAutotuner(
            fn,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            configs=configs,
            save_cache_hook=save_cache_hook,
            mutated_arg_names=mutated_arg_names,
            heuristic_type=heuristic_type,
            size_hints=size_hints,
            custom_kernel=custom_kernel,
        )

    return decorator


######################################################
## Main entry points for triton kernel invocation   ##
## adapts original heuristics for NPU arch, and     ##
## redirect to NPUCaching autotuner                 ##
######################################################

def grid(*numels):
    def grid_fn(meta):
        split_axis_order = meta["split_axis_order"]

        if split_axis_order is not None and  split_axis_order < len(numels) :
            numel = numels[split_axis_order] if split_axis_order is not None else 1
            xblock = meta["XBLOCK"]
            axis2_order = meta["axis2_order"]
            dtype = meta["split_axis_dtype"]
            NBLOCKS, _ = SplitTiling.decide_nblocks_xblock(numel, axis2_order is None, dtype, xblock)
        else:
            NBLOCKS = 1

        log.warning("launch grid(%s), NBLOCKS:%d, meta:%s", numels, NBLOCKS, meta)
        return (
            NBLOCKS,
            1,
            1,
        )

    return grid_fn

# split:sizeof split, xblock:axis1 length, rblock:axis2 length
def triton_config_npu_index(
    size_hints,
    inductor_meta,
    triton_meta=None,
    reduction = False,
    persistent_reduction = False,

) -> List[Config]:
    num_warps = 1
    num_stages = 1
    configs = []

    split_axis_order = inductor_meta["split_axis_order"]
    axis1_order = inductor_meta["axis1_order"]
    axis2_order = inductor_meta["axis2_order"]
    low_dims = inductor_meta["low_dims"]
    split_axis_dtype = inductor_meta["split_axis_dtype"]
    split_numel = size_hints[split_axis_order] if split_axis_order is not None else 1
    is_low_dim = True if split_axis_order is not None and split_axis_order in low_dims else False
    nblocks, split = SplitTiling.decide_nblocks_xblock(split_numel, axis2_order is None, split_axis_dtype)

    log.warning("generating tiling : size_hints:%s split_axis_order:%s, axis1_order:%s, axis2_order:%s, "
                "low_dims:%s  nblocks %s, split:%s persistent_reduction:%s split_axis_dtype:%s", size_hints,
                split_axis_order, axis1_order, axis2_order, low_dims, nblocks, split,
                persistent_reduction, split_axis_dtype)
    # xblock is a range, don't auto_tune
    xnumel = split if split_axis_order == axis1_order else size_hints[axis1_order]
    rblock = 1
    if axis2_order is not None :
        rblock =  split if split_axis_order == axis2_order else size_hints[axis2_order]

    xblock_sub = xnumel
    cfg = {"NBLOCKS": nblocks, "XBLOCK": split, "XBLOCK_SUB": xblock_sub}
    # forward to grid()
    cfg["split_axis_order"] = split_axis_order
    cfg["axis2_order"] = axis2_order
    cfg["is_low_dim"] = is_low_dim
    cfg["split_axis_dtype"] = split_axis_dtype
    is_1d_reduction = reduction and axis2_order is None
    if persistent_reduction :
        numof_reduction_axis = inductor_meta["numof_reduction_axis"]
        if numof_reduction_axis > 1 :
            del cfg["XBLOCK_SUB"]
            configs.append(Config(cfg, num_warps=1, num_stages=1))
        elif axis2_order is None :
            del cfg["XBLOCK"]
            del cfg["XBLOCK_SUB"]
            cfg["NBLOCKS"] = 1
            configs.append(Config(cfg, num_warps=1, num_stages=1))
        else :
            TileGenerator.descend_xblock(rnumel = rblock, xblock=xnumel, configs=configs, cfg=cfg)
    elif is_1d_reduction:
        cfg["NBLOCKS"] = 1
        cfg["XBLOCK"] = split_numel
        cfg["XBLOCK_SUB"] = split_numel
        TileGenerator.descend_xblock(rnumel = rblock, xblock=split_numel, configs=configs, cfg=cfg)
    # both of the two axis are low dims
    elif axis1_order in low_dims and axis2_order in low_dims :
        cfg["RBLOCK"] = rblock
        TileGenerator.descend_xblock_rblock(rnumel = rblock, xblock=xnumel, configs=configs, cfg=cfg)
    elif axis2_order is None and axis1_order is not None:
        TileGenerator.descend_xblock(rnumel=0, xblock=xnumel, configs=configs, cfg=cfg)
    # need to maximize xblock_sub
    elif axis1_order in low_dims:
        cfg["RBLOCK"] = rblock
        TileGenerator.descend_rblock(rnumel = rblock, xblock=xnumel, configs=configs, cfg=cfg)
    elif axis2_order in low_dims:
        cfg["RBLOCK"] = rblock
        TileGenerator.descend_xblock(rnumel=rblock, xblock=xnumel, configs=configs, cfg=cfg)
    else :
        cfg["RBLOCK"] = rblock
        tmp = Config(cfg, num_warps=num_warps, num_stages=num_stages)
        configs.append(tmp)

    for cfg in configs :
        log.warning("generated tiling configs %s", cfg.kwargs)

    return configs
def pointwise_npu_index(
    size_hints,
    triton_meta,
    tile_hint=None,
    filename=None,
    min_elem_per_thread=0,
    inductor_meta=None,
):

    inductor_meta = {} if inductor_meta is None else inductor_meta
    triton_config_with_settings = functools.partial(
        triton_config_npu_index
    )
    return cached_autotune(
        size_hints,
        triton_config_with_settings(size_hints, inductor_meta = inductor_meta),
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.POINTWISE,
        filename=filename,
    )

def reduction_npu_index(
    size_hints,
    reduction_hint=False,
    triton_meta=None,
    filename=None,
    inductor_meta=None,
):

    """args to @triton.heuristics()"""
    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint
    assert triton_meta is not None
    contiguous_config = triton_config_npu_index(size_hints, inductor_meta = inductor_meta, reduction = True)
    return cached_autotune(
        size_hints,
        [
            *contiguous_config,
        ],
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        filename=filename,
        heuristic_type=HeuristicType.REDUCTION,
    )

def persistent_reduction_npu_index(
    size_hints,
    reduction_hint=False,
    triton_meta=None,
    filename=None,
    inductor_meta=None,
):
    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint
    configs = triton_config_npu_index(size_hints, inductor_meta = inductor_meta, reduction=True,
                                      persistent_reduction = True )


    return cached_autotune(
        size_hints,
        configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        filename=filename,
        heuristic_type=HeuristicType.PERSISTENT_REDUCTION,
    )

# fixme , need to add npu_indexing tiling
def foreach(triton_meta, num_warps, filename=None, inductor_meta=None):
    """
    Compile a triton foreach kernel
    """
    return cached_autotune(
        None,
        [triton.Config({}, num_stages=1, num_warps=num_warps)],
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )
