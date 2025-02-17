import logging
import pdb
from typing import List, Tuple, Dict, Any, Optional
from torch._inductor.virtualized import V
from torch._inductor.ir import ( ReductionHint, IRNode, ModularIndexing, FloorDiv)
from torch._inductor.utils import sympy_subs,sympy_index_symbol
from inductor_npu.codegen.triton import NPUIndexTritonKernel
import sympy
import itertools


# NPU doesn't need to support ReductionHint.OUTER, and persistent reduction
def num_splits(
        device,
        dst_dtype,
        src_dtype,
        inner_fn,
        ranges,
        reduction_ranges,
        reduction_type,
        reduction_numel,
        input_node: Optional[IRNode] = None,
    ):
    return ReductionHint.DEFAULT, 1

def detect_flattened_dims(kernel, index):
    new_vars = {}
    if not isinstance(index, sympy.core.add.Add):
        return new_vars
    def detect_flattened_axis(expr):
        def init_new_vars(var, length):
            if var not in new_vars:
                new_vars[var] = {length:[None,None]}
            if length not in new_vars[var] :
                new_vars[var][length] = [None,None]
        if isinstance(expr, ModularIndexing):
            var, divisor, length = expr.args
            init_new_vars(var, length)
            new_vars[var][length][1] = (expr, divisor, length)
        elif isinstance(expr, FloorDiv):
            var, divisor = expr.args
            init_new_vars(var, divisor)
            # over than 1 node_schedule, var may be deleted in kernel.range_tree_nodes
            # it shoule be find in range_tree_nodes_removed dict
            if (var in kernel.range_tree_nodes):
                numel = kernel.range_tree_nodes[var].length
            else:
                numel = kernel.range_tree_nodes_removed[var].length

            length = expr.eval(numel, divisor)
            new_vars[var][divisor][0] = (expr, divisor, length)

        else:
            for x in expr.args:
                detect_flattened_axis(x)

    # add
    for x in index.args:
        detect_flattened_axis(x)

    # make sure FloorDiv, MouldarIndexing must be in-pair
    for var, divisors in new_vars.items():
        if var in kernel.range_tree_nodes:
            parent_axis = kernel.range_tree_nodes[var]
        else:
            parent_axis = kernel.range_tree_nodes_removed[var]
        for divisor, pair in divisors.items():
            if not pair[0] and not pair[1] :
                pass
            #FloorDiv not inplace
            elif not pair[0] :
                _, _, length = pair[1]
                expr = FloorDiv(var, length)
                new_vars[var][divisor][0] = (expr, length, parent_axis.length //length )
            #ModularIndexing not inplace
            elif not pair[1] :
                expr = ModularIndexing(var, 1, divisor)
                new_vars[var][divisor][1] = (expr, 1, divisor)
            else :
                pass

    return new_vars

def rebuild_flattened_dims(indexing) :
    def rebuild_flattened_dim(key, index,  old_node, flatten_dim) :
        for _, pair in flatten_dim.items():
            new_var_expr = sympy.Integer(0)
            origin_axis_length = 0
            for axis in pair:
                expr, divisor, length = axis
                # 3. try to rebuild the axis in kernel
                new_node = old_node.parent.lookup(divisor, length)
                # 4. substitute div/mod expression in indexing
                index = index.subs(expr, new_node.symbol())
                indexing[key] = index
                if isinstance(expr, FloorDiv):
                    new_var_expr = new_var_expr + new_node.symbol() * divisor
                    origin_axis_length = divisor * length
                elif isinstance(expr, ModularIndexing):
                    new_var_expr = new_var_expr + new_node.symbol()
                V.kernel.expr_substituted[expr] = new_node.symbol()

            if var not in  V.kernel.range_tree_nodes_substituted :
                V.kernel.range_tree_nodes_substituted[var] = []
            V.kernel.range_tree_nodes_substituted[var].append((origin_axis_length,new_var_expr))

    def find_index_in_substitute(index, kernel) :
        return any([index.find(key) for key in kernel.expr_substituted.keys()])

    kernel = V.kernel
    for key, index in indexing.items():
        if find_index_in_substitute(index, kernel):
            new_index = sympy_subs(index, kernel.expr_substituted)
            indexing[key] = new_index

        # 1. try to find out flattened axis from indexing
        flatten_dims = detect_flattened_dims(kernel, index)
        #2. try to rebuild these flattened dims
        for var, flatten_dim in flatten_dims.items():
            if (var in kernel.range_tree_nodes):
                old_node = kernel.range_tree_nodes[var]
            else:
                old_node = kernel.range_tree_nodes_removed[var]

            rebuild_flattened_dim(key, index, old_node, flatten_dim)

def substituted_dims_in_indexing(self, indexing, kernel, range_tree_nodes_substituted) :
    substituted = False
    for var, candidates in range_tree_nodes_substituted.items():
        assert len(candidates) > 0, candidates
        exprs = sorted(candidates, reverse=True, key=lambda x: x[0])
        # the best candidate is with the longest numel
        numel = exprs[0][0]
        expr = exprs[0][1]
        node = kernel.range_tree_nodes[var]
        if node.length != numel:
            logging.warning("sub nodes (expr%s, numel:%d) can not substitute parent node(%s:%d)",
                            expr, numel, node.symbol(), node.length)
            continue
        for key, index in indexing.items():
            if var in index.free_symbols:
                index = index.subs(var, expr)
                indexing[key] = index
                substituted = True

    return substituted

def transform_dims_in_indexing(self, indices) :
    if self.indexing is None :
        index = list(itertools.chain.from_iterable(indices))
        assert len(index) == len(self.var_ranges), (index, self.var_ranges)
        assert all(v not in self.var_ranges for v in index)
        replacements = dict(zip(self.var_ranges.keys(), index))
        indexing_map = dict(zip( index, self.var_ranges.keys()))
        setattr(self, 'indexing_map', indexing_map)
        self.indexing = {
            name: sympy_subs(expr, replacements)
            for name, expr in self.indexing_exprs.items()
        }

    if V.kernel is not None and isinstance(V.kernel, NPUIndexTritonKernel):
        rebuild_flattened_dims(self.indexing)

# select tiling axis, recover missing dimensions,
def loopbody__call__(self, *indices):
    if self.indexing is None:
        index = list(itertools.chain.from_iterable(indices))
        assert len(index) == len(self.var_ranges), (index, self.var_ranges)
        assert all(v not in self.var_ranges for v in index)
        replacements = dict(zip(self.var_ranges.keys(), index))
        self.indexing = {
            name: sympy_subs(expr, replacements)
            for name, expr in self.indexing_exprs.items()
        }
    result = self.root_block()
    self.indexing = None
    return result


