import pdb

from inductor_npu.codegen.triton import NPUIndexTritonKernel
from torch._inductor.codegen.triton import ( TritonScheduling, log, config, EnableReduction, DisableReduction,
                                             indexing_dtype_strength_reduction)
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.virtualized import (
    V,
)
from torch._inductor.codecache import code_hash
from torch._dynamo.utils import counters
import itertools, contextlib
from torch._inductor.utils import sympy_index_symbol,ModularIndexing,FloorDiv
import sympy
from .split_tiling import SplitTiling

class NPUTritonScheduling(TritonScheduling):
    def __init__(self, scheduler):
        super().__init__(scheduler)
        self.kernel_type = NPUIndexTritonKernel

    # create NPUTritonKernel or NPUIndexTritonKernel
    # set final_kernel to V after kernel context exits
    def codegen_node_schedule(
            self, node_schedule, buf_accesses, numel, reduction_numel
        ):
        from torch._inductor.codegen.triton_split_scan import TritonSplitScanKernel
        tiled_groups = self.select_tiling(node_schedule, numel, reduction_numel)
        reduction_hint_val, mutations, index_dtype = self.get_kernel_args(
            node_schedule, numel, reduction_numel
        )

        is_split_scan = any(
            isinstance(node, BaseSchedulerNode) and node.is_split_scan()
            for node in node_schedule
        )
        # Note: backported patch
        kernel_type = TritonSplitScanKernel if is_split_scan else self.kernel_type
        kernel_args = tiled_groups
        kernel_kwargs = {
            "reduction_hint": reduction_hint_val,
            "mutations": mutations,
            "index_dtype": index_dtype,
        }
        kernel = kernel_type(
            *kernel_args,
            **kernel_kwargs,
        )
        kernel.buf_accesses = buf_accesses
        setattr(kernel, "node_schedule", node_schedule )
        # generate code for the kernel
        self.decide_codegen_dims_in_kernel(node_schedule, kernel)
        self.codegen_node_schedule_with_kernel(node_schedule, kernel)

        with V.set_kernel_handler(kernel):
            src_code = kernel.codegen_kernel()

        kernel_name = self.define_kernel(src_code, node_schedule)
        log.debug("Generating kernel code with kernel_name: %s", kernel_name)
        kernel.kernel_name = kernel_name
        kernel.code_hash = code_hash(src_code)
        #NPU don't need persistent reduction
        final_kernel = kernel  # type: ignore[assignment]
        with V.set_kernel_handler(final_kernel):
            for node in node_schedule:
                if node not in (EnableReduction, DisableReduction):
                    node.mark_run()
        setattr(V, "final_kernel", final_kernel)
        self.codegen_comment(node_schedule)
        final_kernel.call_kernel(final_kernel.kernel_name)
        if config.nan_asserts:
            final_kernel.codegen_nan_check()
        if config.warn_mix_layout:
            final_kernel.warn_mix_layout(kernel_name)

        V.graph.removed_buffers |= final_kernel.removed_buffers
        V.graph.inplaced_to_remove |= final_kernel.inplaced_to_remove

        if (
                V.graph.wrapper_code.supports_intermediate_hooks
                and config.generate_intermediate_hooks
        ):
            # Not every node in the schedule will actually be live on output;
            # we can't check dead buffers.
            live_outs = kernel.args.live_output_buffers()
            for node in node_schedule:
                if not isinstance(node, BaseSchedulerNode):
                    continue
                name = node.get_name()
                if name not in live_outs:
                    continue
                origin_node = node.node.get_origin_node()
                if origin_node is not None:
                    counters["inductor"]["intermediate_hooks"] += 1
                    V.graph.wrapper_code.writeline(
                        f"run_intermediate_hooks({origin_node.name!r}, {name})"
                    )

        self.scheduler.free_buffers()

    def decide_codegen_dims_in_kernel(self, node_schedule, kernel):
        def current_reduction_nodes(nodes):
            return itertools.takewhile(lambda n: n is not DisableReduction, nodes)

        with kernel:
            # 1. transform dims: create new dims to substitute floor_divide and modular expression
            stack = contextlib.ExitStack()
            for  i, node in enumerate(node_schedule):
                if node is DisableReduction:
                    stack.enter_context(kernel.disable_reduction())
                elif node is EnableReduction:
                    stack.close()
                    kernel.set_last_usage(current_reduction_nodes(node_schedule[i:]))
                else:
                    index_vars = kernel.split_and_set_ranges(node.get_ranges())
                    node._body.transform_dims_in_indexing(index_vars)

            # 2.collection additional node to be substituted
            self.additional_nodes_to_be_subs(kernel, kernel.range_tree_nodes_substituted)
            # 3.do the substitution on all indexing
            for  node in node_schedule:
                if node  in (EnableReduction, DisableReduction):
                    continue
                indexing = node._body.indexing
                node._body.substituted_dims_in_indexing(indexing, kernel, kernel.range_tree_nodes_substituted)
            # 4.remove the substituted dims from kernel
            for var, _ in kernel.range_tree_nodes_substituted.items():
                if (var in kernel.range_tree_nodes):
                    root = kernel.range_tree_nodes[var].parent
                    root.remove_entry(var)
            # select split and tiling axis
            split_tiling = SplitTiling(kernel)
            split_tiling.select_tiling_axis()
            # debug print index transforms
            # for node in node_schedule:
            #   if node in (EnableReduction, DisableReduction):
            #       continue
            #   for x,y in zip( node._body.indexing_exprs.values(), node._body.indexing.values()) :
            #       print(f"index transform:{x}->{y}")

    def additional_nodes_to_be_subs(self, kernel, node_to_be_substituted):
        for node in kernel.range_tree_nodes.values():
            if node.expr != sympy_index_symbol(f"{node.parent.prefix}index") \
                    or len(node.parent.var_ranges) == 1 \
                    or node.symbol() in node_to_be_substituted:
                continue
            numel = sympy.Integer(1)
            new_var_expr = sympy.Integer(0)
            for k, s in node.parent.var_ranges.items():
                if k == node.symbol():
                    continue
                numel = numel * s
                sub_node = kernel.range_tree_nodes[k]
                if isinstance(sub_node.expr, FloorDiv):
                    new_var_expr = new_var_expr + sub_node.symbol() * sub_node.divisor
                elif isinstance(sub_node.expr, ModularIndexing):
                    new_var_expr = new_var_expr + sub_node.symbol()

            if numel == node.length:
                node_to_be_substituted[node.symbol()] = [(node.length, new_var_expr)]
            else:
                log.warning("sub nodes (expr%s, numel:%d) can not make up parent node(%s:%d)",
                                new_var_expr, numel, node.symbol(), node.length)





