import pdb

import torch
from torch._inductor.utils import sympy_subs
from torch._inductor.scheduler import SchedulerNode
from typing import List,Set,Iterable,Callable
import sympy
import operator
import itertools

from torch._inductor.codegen.triton import (
    IndexingOptions,
    sympy_dot,
    CantSplit,
    triton_reshape,
    TritonCSEVariable,
    free_symbol_startswith,
    OpsHandler, DisableReduction, EnableReduction,
)

from torch._inductor.codegen.triton import (

    TritonKernel,
    TritonKernelOverrides,
    IterationRangesRoot,
    IterationRangesEntry,
    CSEVariable,
    gen_common_triton_imports,
    ReductionHint,
    BlockPtrOptions,
    triton_acc_type,
    triton_constant,
    is_welford_reduction,
    triton_compute_type,
    cast,
    ModularIndexing, FloorDiv ,sympy_index_symbol,
    log
)

from torch.utils import _pytree as pytree
from torch.utils._sympy.value_ranges import ValueRanges

from typing import Dict
from enum import Enum
import functools

from torch._inductor import config, ir
from torch._inductor.virtualized import (
    V,
    StoreMode,
    ReductionType,
    _ops as ops,
)

from torch._inductor.utils import (
    Placeholder,
    next_power_of_2,
)


from torch._inductor.codegen.common import (
    IndentedBuffer,
    SizeArg,
    DeferredLine,
)
from torch._inductor.codegen.triton_utils import config_of, signature_of, signature_to_meta

from typing import (
    Optional,
    Union,
    Tuple,
    Any,
)
import re

def flatten(nums):
    res = []
    for i in nums:
        if isinstance(i, list):
            res.extend(flatten(i))
        else:
            res.append(i)
    return res

class AxisDirection(Enum):
    Flat = 0,
    Vertical = 1,
    Horizontal = 2

def reverse_direction(direction):
  if direction == AxisDirection.Vertical :
      return AxisDirection.Horizontal
  elif direction == AxisDirection.Horizontal :
      return AxisDirection.Vertical
  else :
       return AxisDirection.Flat


class NPUTritonKernelOverrides(TritonKernelOverrides):
    @staticmethod
    def exp(x):
        return f"libdevice.exp({x})"
    @staticmethod
    def rsqrt(x):
        return f"tl.rsqrt({x})"
    @staticmethod
    def floor(x):
        return f"tl_math.floor({x})"

    @staticmethod
    def ceil(x):
        return f"tl_math.ceil({x})"
class NumelList(Tuple):

    def numels(self):
        numel = functools.reduce(lambda a, b: a * b, self)
        return numel

    def __eq__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel == numel2

    def __mod__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel % numel2

    def __truediv__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel / numel2
    def __floordiv__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel // numel2

    def __mul__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel * numel2
    def __rmul__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel * numel2

    def __add__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel + numel2
    def __radd__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel + numel2

    def __hash__(self):
        return super(NumelList, self).__hash__()

def group_fn(self, sizes):
    groups = list()
    for s in sizes :
       if not s :
           groups.append(1)
       elif isinstance(s, list):
           group = flatten(s)
           groups.append(NumelList(tuple(group)) if isinstance(group, list) else group)
       else :
           groups.append(s)
    return tuple(groups)

@staticmethod
def select_index_dtype(node_schedule, numel, reduction_numel):
    return "tl.int32"

@classmethod
def select_tiling(cls, node_schedule, numel, reduction_numel=sympy.Integer(1)):
    return (numel, reduction_numel)

class IterationRangesEntryNPUIndex(IterationRangesEntry) :
    def __init__(
            self,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_tiling_axis1 = False
        self.is_tiling_axis2 = False
        self.is_split_axis = False
        self.indexing_code = IndentedBuffer()
        self.sorted_order = None
        self.low_dims = set()


    def _codegen_mask(self):
        if self.is_tiling_axis1 or self.is_tiling_axis2 :
            upper = f"{self.name}_numel"
            line = f"{self.name}_mask = {self.name} < {upper}"
            self.writeline(line)
            line = f"{self.name}_prime_mask = {self.name}_prime < {upper}"
            self.writeline(line)
        else:
            pass

    def _codegen(self):
        index = None
        vertical = self.is_tiling_axis1 if V.kernel.numof_reduction_axis() <=1 else not isinstance(self.expr, ModularIndexing)
        direction = V.kernel.get_axis_direction(vertical)
        # for multiple reduce dims, don't need this
        if self.is_tiling_axis1 and  V.kernel.numof_reduction_axis() <= 1:
            index = f"{self.name} = {self.codegen_index(direction)}"
            #to be fixed, only permute need to this .
            self.writeline(f"{self.name}_prime = {self.codegen_index(reverse_direction(direction))}")

        elif self.is_tiling_axis2:
            index = f"{self.name} = {self.codegen_index(direction)}"
            #to be fixed, only permute need to this .
            self.writeline(f"{self.name}_prime = {self.codegen_index(reverse_direction(direction))}")
            if V.kernel.inside_reduction and V.kernel.current_node  \
                    and isinstance(V.kernel.current_node, SchedulerNode) \
                    and V.kernel.current_node.node \
                    and V.kernel.current_node.node.data \
                    and isinstance(V.kernel.current_node.node.data, ir.Reduction):
                reduction_type = V.kernel.current_node.node.data.reduction_type
                if reduction_type in {"argmax", "argmin"} :
                    self.writeline(f"{self.parent.prefix}index = "
                                   f"{self.codegen_index(reverse_direction(AxisDirection.Flat))}")
        if index:
            self.writeline(index)
            self._codegen_mask()
        return self.name
    def writeline(self, line):
        self.indexing_code.writeline(line)

    def codegen_index(self, direction):
        if self.is_tiling_axis1 and V.kernel.axis2 is None and V.kernel.persistent_reduction:
            index = f"tl.arange(0, RBLOCK)"
            return index
        elif self.is_tiling_axis1:
            if self.is_split_axis :
                offset = f"{self.symbol()}_offset"
                index = f"{offset} + (loop1 * XBLOCK_SUB) + base1"
            else :
                index = f"(loop1 * XBLOCK_SUB) + base1"

            if V.kernel.axis2 is not None and direction != AxisDirection.Flat :
                index += ("[None, :]" if direction == AxisDirection.Horizontal else "[:, None]")
            return index
        elif self.is_tiling_axis2  :
            if V.kernel.persistent_reduction :
                index = f"tl.arange(0, RBLOCK_{self.symbol()})" if V.kernel.numof_reduction_axis() > 1 else  "base2"
            else :
                index = "loop2 * RBLOCK + base2"

            if direction != AxisDirection.Flat:
                index += ("[:, None]" if direction == AxisDirection.Vertical else "[None, :]")
            return index

    def codegen_header(self, code):
        # generate offset index loop
        lines = []

        if self.is_split_axis and not (V.kernel.axis2 is None and V.kernel.persistent_reduction):
            lines.append(f"{self.symbol()}_offset = tl.program_id(0) * XBLOCK")

        if self.is_tiling_axis1 and not (V.kernel.axis2 is None and V.kernel.persistent_reduction):
            #  don't create loops for multi-reductions
            if V.kernel.numof_reduction_axis() <= 1 :
                lines.append("base1 = tl.arange(0, XBLOCK_SUB)")
                xblock = f"XBLOCK" if self.is_split_axis else f"{self.symbol()}_numel"
                lines.append(f"loops1 = ({xblock} + XBLOCK_SUB - 1) // XBLOCK_SUB")

        elif self.is_tiling_axis2 and  len(V.kernel.axis2_list) <=1:
            lines.append("base2 = tl.arange(0, RBLOCK)")
            lines.append(f"loops2 = ({self.name}_numel + RBLOCK - 1) // RBLOCK" )
        else:
            pass

        code.writelines(lines)

class IterationRangesRootNPUIndex(IterationRangesRoot):
    def __init__(
            self,
            name: str,
            numel: sympy.Expr,
            prefix: str,
            index: int,
            kernel: TritonKernel,
            pid_cache=None,
            *,
            is_loop: bool,
            tensor_dim: Optional[int],
            grid_dim: Optional[int],
    ):
        super().__init__(name, numel, prefix, index, kernel, pid_cache, is_loop=is_loop, tensor_dim=tensor_dim,
                         grid_dim=grid_dim)

    def __repr__(self):
        return f"IterationRangesRootNPUIndex({self.name!r}, {self.numel}, ...)"

    def remove_entry(self, name):
        if name in self.var_ranges :
            del self.var_ranges[name]
        if name in self.var_list:
            del self.var_list[self.var_list.index(name)]
        if name in V.kernel.range_tree_nodes :
            V.kernel.range_tree_nodes_removed[name] = V.kernel.range_tree_nodes[name]
            del V.kernel.range_tree_nodes[name]
        if name in self.nodes:
            del self.nodes[name]

    def lookup(self, divisor, length):
        """
        Lookup a given RangeTreeEntry, creating it if needed
        """
        if V.graph.sizevars.statically_known_equals(divisor * length, self.numel):
            expr = FloorDiv(sympy_index_symbol(f"{self.prefix}index"), divisor)
        else:
            expr = ModularIndexing(
                sympy_index_symbol(f"{self.prefix}index"), divisor, length
            )

        if expr not in self.nodes:
            node = IterationRangesEntryNPUIndex(
                f"{self.prefix}{next(V.kernel.iter_vars_count)}",
                divisor,
                length,
                expr,
                self,
            )
            V.kernel.range_tree_nodes[node.symbol()] = node
            self.var_list.append(node.symbol())
            self.var_ranges[node.symbol()] = length
            self.nodes[expr] = node


        return self.nodes[expr]


def is_compatible(groups: Iterable[sympy.Expr], lengths: List[List[sympy.Expr]]):
    try:
        groups = flatten(groups)
        NPUIndexTritonKernel._split_iteration_ranges(groups, lengths)
        return True
    except CantSplit:
        return False

class NPUIndexTritonKernel(TritonKernel):
    overrides = NPUTritonKernelOverrides

    def __init__(self,
        *groups,
        index_dtype: str,
        mutations: Optional[Set[str]] = None,
        pid_cache=None,
        reduction_hint=ReductionHint.DEFAULT,
        min_elem_per_thread=0,
        disable_persistent_reduction=False,):

        super().__init__(*groups, index_dtype=index_dtype, mutations=mutations, pid_cache = pid_cache,
                         reduction_hint=reduction_hint, min_elem_per_thread = min_elem_per_thread,
                         disable_persistent_reduction = disable_persistent_reduction )
        self.first_node = True
        self.inside_high_order_reduction = False
        # split axis
        self.split_axis = None
        # tiling axis
        self.axis1 = None
        self.axis2 = None
        # incase two reduction axis
        self.axis2_list = []
        self.low_dims  = set()

        self.range_tree_nodes_removed: Dict[sympy.Symbol, IterationRangesEntry] = {}
        self.range_tree_nodes_substituted = {}
        self.expr_substituted = {}
        self.sorted_axis = []
        self.prefix: IndentedBuffer = IndentedBuffer()

    def gen_triton_ext_imports(self):
        imports = IndentedBuffer()
        imports.splice(
            """
            from torch._inductor import triton_helpers
            from inductor_npu import npu_triton_heuristics
            from inductor_npu import npu_triton_helpers
            from inductor_npu.npu_triton_helpers import libdevice, math as tl_math
            import torch
            """
        )
        return imports.getvalue()

    def patch_triton_hash(self):
        # remove this method once the original invocation is fixed
        import hashlib
        from triton.compiler.compiler import triton_key, make_backend
        from triton.runtime.driver import driver
        backend = make_backend(driver.active.get_current_target())
        key = f"{triton_key()}-{backend.hash()}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    # persistent_reduction means reduction without loop2
    # for big reduction numel (> 1024), should use outer reduction or loop2 inner reduction
    def should_use_persistent_reduction(self) -> bool:
        if not (self.inside_reduction and config.triton.persistent_reductions):
            return False
        threshold = {
            ReductionHint.INNER: 1024,
            ReductionHint.DEFAULT : 1024
        }.get(self.reduction_hint, 64)

        if config.triton.multi_kernel:
            threshold *= 16
        last_numel = self.numels[-1]
        if isinstance(last_numel, (list, NumelList)) :
            last_numel = NumelList(last_numel).numels()
            self.numels[-1] = last_numel

        if not isinstance(last_numel, (int, sympy.Integer)):
            # Not static
            return False
        hint = V.graph.sizevars.size_hint(last_numel)
        if hint > threshold:
            return False
        # will need to recompile if we cross a larger power of 2 boundary
        V.graph.sizevars.guard_leq(last_numel, next_power_of_2(hint))  # type: ignore[arg-type]
        return True

    def numof_reduction_axis(self):
        root = self.range_trees[-1]
        if root is None :
            return 0

        return len(root.var_list)

    def numof_tiling_axis(self):
        return  (1 if self.axis1 is not None else 0) + (1 if self.axis2 is not None else 0 )

    def initialize_range_tree(self, pid_cache):
        self.numels = flatten(self.numels)
        self.total_numels = 0
        for x in self.numels :
            if not isinstance(x, sympy.Integer) :
                x = x.subs(V.graph.sizevars.var_to_val)
            if x > 1 :
                self.total_numels +=1
        #self.select_tiling_axis()
        no_r_dim = not self.inside_reduction or self.numels[-1] == 1
        prefixes = "wvtpyxr"
        active_prefixes = prefixes[-len(self.numels) :]
        #prefix can not be 's', 'u', 'ps' , 'i', 'z', 'q'
        grid_dims = "xyptvw"
        if self.no_x_dim:
            tensor_dims = "r"
        elif no_r_dim:
            tensor_dims = "xyptvw"
        else:
            tensor_dims = "xyptvwr"
        tensor_dims = "".join(p for p in tensor_dims if p in active_prefixes)
        for i, prefix in enumerate(active_prefixes):
            is_reduction = prefix == "r"
            tensor_dim = tensor_dims.find(prefix) if prefix in tensor_dims else None
            grid_dim = None if is_reduction else grid_dims.find(prefix)
            index = i if grid_dim is None else grid_dim
            self.range_trees.append(
                IterationRangesRootNPUIndex(
                    f"{prefix}index",
                    self.numels[i],
                    prefix,
                    index,
                    self,
                    pid_cache=pid_cache,
                    is_loop=is_reduction and not self.persistent_reduction,
                    tensor_dim=tensor_dim,
                    grid_dim=grid_dim
                )
            )

    # numels sent to autotune configs
    def get_size_hints(self):
        size_hints = []

        if (len(self.range_tree_nodes.values()) == 0):
            return self.numels
        for i, node in enumerate(self.sorted_axis):
            if isinstance(node.expr, ModularIndexing):
                numel_expr = node.length
            else:
                numel_expr = node.expr.subs({sympy_index_symbol(r.name): r.numel for r in self.range_trees})

            numel_expr = V.graph.sizevars.symbolic_hint(numel_expr)

            size_hints.append(numel_expr)
        return size_hints
    def add_numel_to_call_args_and_grid(self, name, call_args, grid):
        for node in self.sorted_axis:
            if isinstance(node.expr, ModularIndexing) :
                numel_expr = node.length
            else :
                numel_expr = node.expr.subs({sympy_index_symbol(r.name): r.numel for r in self.range_trees})

            if isinstance(numel_expr, (sympy.Integer, sympy.Symbol)):
                expr = numel_expr
            else:
                expr = V.graph.wrapper_code.generate_node_numel_expr(name, node, numel_expr)
            call_args.append(expr)
            if node.parent.grid_dim is not None:
                grid.append(expr)

    def gen_numel_args(self, signature, triton_meta_signature, argdefs ):
        for node in self.sorted_axis:
            sizearg = SizeArg(f"{node.name}_numel", node.length)
            signature.append(sizearg)
            triton_meta_signature[len(argdefs)] = signature_of(
                sizearg, size_dtype=self.index_dtype
            )
            argdefs.append(f"{node.name}_numel")

    def codegen_kernel(self, name=None):
        code = IndentedBuffer()
        size_hints = self.get_size_hints()
        heuristics = self._get_heuristic()
        if name is None:
            code.splice(gen_common_triton_imports())
            # Note: add extra imports for extensions
            code.splice(self.gen_triton_ext_imports())

            if config.benchmark_kernel:
                code.splice(self.imports_for_benchmark_kernel())

        argdefs, _, signature = self.args.python_argdefs()
        for i, arg in enumerate(signature):
            if isinstance(arg, SizeArg):
                symbol = cast(sympy.Symbol, arg.expr)
                if symbol in V.graph.sizevars.inv_precomputed_replacements:
                    signature[i] = SizeArg(
                        arg.name, V.graph.sizevars.inv_precomputed_replacements[symbol]
                    )

        triton_meta_signature = signature_to_meta(
            signature, size_dtype=self.index_dtype
        )
        triton_meta = {
            "signature": triton_meta_signature,
            "device": V.graph.scheduler.current_device.index,
            "device_type": V.graph.scheduler.current_device.type,
            "constants": {},
            # special config for NPU, specify compile target
            "mix_mode": "aiv",
        }

        inductor_meta = self.create_inductor_meta()
        num_gb = None
        if config.benchmark_kernel or config.profile_bandwidth:
            num_gb = self.estimate_kernel_num_bytes() / 1e9
            inductor_meta["kernel_num_gb"] = num_gb
        self.gen_numel_args(signature, triton_meta_signature, argdefs)
        #triton_meta["configs"] = [config_of(signature)]

        self.triton_meta = triton_meta
        #add in tiling args
        self.add_autotune_args(argdefs)
        #for scalar codegen
        if len(self.range_tree_nodes) == 0:
            self.write_scalar()
        else:
            self.codegen_body()
        # from ..codegen.triton_utils import npu_indexing_config_of
        # inductor_meta["configs"] = [npu_indexing_config_of(signature, size_hints)]

        for helper in self.helper_functions:
            code.writeline("")
            code.splice(helper)


        # Note: override original triton_heuristics
        if self.inside_reduction:
            reduction_hint = self.reduction_hint
            heuristics_line = f"""
                @npu_triton_heuristics.{heuristics}(
                    size_hints={size_hints},
                    reduction_hint={reduction_hint},
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r}
                )
                @triton.jit
            """
        else:
            tile_hint = ""
            if len(size_hints) == 2:
                if len(signature) == 4:  # input, output and 2 args
                    tile_hint = "tile_hint=TileHint.SQUARE,"
                else:
                    tile_hint = "tile_hint=TileHint.DEFAULT,"
            heuristics_line = f"""
                @npu_triton_heuristics.{heuristics}(
                    size_hints={size_hints!r}, {tile_hint}
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r},
                    min_elem_per_thread={self.min_elem_per_thread}
                )
                @triton.jit
            """
        code.splice(heuristics_line)
        code.writeline(
            f"def {name or str(Placeholder.KERNEL_NAME)}({', '.join(argdefs)}):"
        )
        with code.indent():
            self.codegen_static_numels(code)
            for old, new in self.args.aliases():
                code.writeline(f"{old} = {new}")
            code.splice(self.body)

        if config.benchmark_kernel:
            code.splice(self.codegen_kernel_benchmark(num_gb))

        return code.getvalue()
    def codegen_static_numels(self, code):
        no_x_axis =  self.numof_reduction_axis() > 1
        symbols = []
        if self.axis2 is not None :
            symbols = list(self.axis2_list) if no_x_axis else list([self.axis2])
        elif self.persistent_reduction and self.axis1 is not None:
            symbols = list([self.axis1])

        nodes = [self.range_tree_nodes[symbol] for symbol in symbols if symbol is not None]
        for node in nodes:
            if node.prefix == "r" and self.persistent_reduction:
                simplified_tree_numel = V.graph.sizevars.simplify(node.length)
                if isinstance(simplified_tree_numel, (sympy.Integer, int)):
                    val = int(simplified_tree_numel)
                else:
                    continue
                val = next_power_of_2(val)
                if no_x_axis :
                    code.writeline(f"RBLOCK_{node.symbol()}: tl.constexpr = {val}")
                else :
                    code.writeline(f"RBLOCK: tl.constexpr = {val}")

    def axis2_variable(self):
        if self.axis2 is not None :
            return self.range_tree_nodes[self.axis2]
        return None

    def is_isolated_symbol(self, input_str, symbol):
        # 使用正则表达式查找独立的符号, 防止out_ptr0 匹配上r0  r0_prime
        pattern1 = r'\b' + re.escape(symbol) + r'\b'
        pattern2 = r'\b' + re.escape(symbol+'_prime') + r'\b'

        return bool(re.search(pattern1, input_str)) or bool(re.search(pattern2, input_str))

    def find_axis2_in_load_store(self):
        var = self.axis2_variable()
        if not var :
            return False
        for line in self.loads._lines :
            if line.find('tl.load') >= 0 and self.is_isolated_symbol(line, var.name):
                return True
        for line in self.compute._lines :
            if line.find('tl.load') >= 0 and self.is_isolated_symbol(line, var.name):
                return True
        for line in self.suffix._lines :
            if line.find('tl.store') >= 0 and self.is_isolated_symbol(line, var.name):
                return True
        for line in self.stores._lines :
            if isinstance(line,DeferredLine) :
                line = line.line
            if line.find('tl.store') >= 0 and self.is_isolated_symbol(line, var.name):
                return True
        return False

    def find_axis2_in_indexing(self):
        var = self.axis2_variable()
        if not var :
            return False
        if self.current_node is None :
            return False
        for index in self.current_node._body.indexing.values() :
            if var.symbol() in index.free_symbols :
                return True
        return False

    def write_scalar(self):
        self.body.splice(self.indexing_code)
        self.body.splice(self.loads)
        self.body.splice(self.compute)
        self.body.splice(self.stores)
        self.loads.clear()
        self.compute.clear()
        self.stores.clear()
        self.suffix.clear()
        self.prefix.clear()

    def codegen_body(self):
        if not (
                self.loads
                or self.stores
                or self.compute
                or self.suffix
        ):
            return

        def write_pointwise() :
            self.body.splice(self.indexing_code)
            self.body.splice(self.loads)
            self.body.splice(self.compute)
            self.body.splice(self.stores)
        def is_1d_reduction() :
            return self.numels[-1] > 1 and self.axis2 is None
        def codegen_range(index) :
            def loop_body(index, indexing_code, is_last_axis, do_indent = True ) :
                if do_indent:
                    self.body.do_indent()
                if indexing_code :
                    self.body.splice(indexing_code)

                if is_last_axis:
                    write_pointwise()
                else:
                    codegen_range(index + 1)

                if do_indent :
                    self.body.do_unindent()

            if index < 0 or index >= len(self.range_tree_nodes):
                return
            nodes = self.sorted_axis
            range = nodes[index]
            is_tilling_asix1 = getattr(range, "is_tiling_axis1")
            is_tilling_asix2 = getattr(range, "is_tiling_axis2")
            is_last_axis = index == len(nodes) -1
            indexing_code = getattr(range, "indexing_code")
            numof_axis2 = self.numof_reduction_axis()
            if is_tilling_asix1:
                do_indent = True
                # 1D reduction or 1d persistent reduction, axis1 is reduction axis
                if is_1d_reduction():
                    self.body.splice(self.prefix)
                    self.prefix.clear()
                
                # multi-dim reduction, i.e. var_mean[1,2]
                if numof_axis2 > 1:
                    if range.is_split_axis :
                        offset = f"{range.name}_offset"
                        self.body.writeline(f"for {range.name} in range({offset}, "
                                             f"{offset} + XBLOCK):")
                    else :
                        self.body.writeline(f"for {range.name} in  range({range.name}_numel):")
                # 1D persistent_reduction or 1d reduction non-first-node
                elif self.axis2 is None and (self.persistent_reduction or len(self.loads._lines) == 0):
                    do_indent = False
                    if len(self.loads._lines) == 0:
                        indexing_code = None
                else :
                    self.body.writeline(f"for loop1 in range(loops1):")
                
                loop_body(index, indexing_code, is_last_axis, do_indent = do_indent)
                
                # for 1D reduction, need to add in suffix for persist_reduction or second node of 1d reduction
                if is_1d_reduction() :
                    self.body.splice(self.suffix)
                    self.suffix.clear()

            elif is_tilling_asix2:
                do_indent = False
                need_axis2_loop = self.find_axis2_in_load_store()
                if not need_axis2_loop :
                    indexing_code = None
                if (not self.inside_reduction or not self.persistent_reduction) \
                        and need_axis2_loop:
                    self.body.splice(self.prefix)
                    self.body.writeline(f"for loop2 in range(loops2):")
                    do_indent = True
                loop_body(index, indexing_code, is_last_axis, do_indent)
                self.body.splice(self.suffix)
                self.suffix.clear()

            elif is_last_axis and range.numel == 1: #pointwise , last axis =1
                write_pointwise()
            else:
                if range.is_split_axis :
                    offset = f"{range.symbol()}_offset"
                    self.body.writeline(f"for {range.symbol()} in range({offset}, {offset} + XBLOCK):")
                else :
                    self.body.writeline(f"for {range.symbol()} in range({range.name}_numel):")
                loop_body(index, indexing_code, is_last_axis)

        if self.first_node:
            for node in self.sorted_axis:
                node.codegen_header(self.body)


        if self.first_node:
            codegen_range(0)
        else :
            if self.axis2 is None :
                codegen_range(0)
            else :
                axis2_order = self.range_tree_nodes[self.axis2].sorted_order
                if self.persistent_reduction and self.numof_reduction_axis() > 1 :
                    axis2_order = axis2_order - self.numof_reduction_axis() +1
                for _ in range(axis2_order) :
                    self.body.do_indent()
                codegen_range(axis2_order)
                for _ in range(axis2_order) :
                    self.body.do_unindent()

        self.cse.invalidate(self.outside_loop_vars)
        self.loads.clear()
        self.compute.clear()
        self.stores.clear()
        self.suffix.clear()
        self.prefix.clear()
        #for root in self.range_trees:
        #    root.cache_clear()
        self.first_node = False

    # for creat constant tensor, if have two axis, constant=tl.full([1,1]) else  tl.full([1])
    def triton_tensor_ndim(self):
        if self.numof_reduction_axis() > 1 :
            return 1
        if self.axis1 is not None and self.axis2 is not None:
            ndim = 2
        else:
            ndim = 1
        return ndim

    # fixme, indexing.mask_str is None , see varmean_test.py
    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable):
        assert self.inside_reduction
        self.inside_reduction = False
        indexing = self.indexing(index, block_ptr=True)
        self.inside_reduction = True
        var = self.args.output(name)
        if isinstance(indexing, BlockPtrOptions):
            self.suffix.writeline(
                DeferredLine(
                    name,
                    self.codegen_block_ptr_store_line(
                        name,
                        indexing,
                        indexing.format(var),
                        value,
                        f", boundary_check={indexing.boundary_check()!r}",
                    ),
                )
            )
        else:
            assert isinstance(indexing, IndexingOptions)
            line = f"tl.store({var} + ({indexing.index_str} ), {value}, {indexing.mask_str})"
            if self.numof_reduction_axis() > 1 :
                line = f"tl.store({var} + ({indexing.index_str} + tl.arange(0,1) ), {value}, {indexing.mask_str})"
            self.suffix.writeline(
                DeferredLine( name, line )
            )

    def apply_var_prime(self, index, line, mask):
        # axis should only be replaced once
        axis_list = []
        for key in index.as_coefficients_dict().keys():
            if not key.free_symbols :
                continue
            symbol = list(key.free_symbols)[0]
            if symbol not in self.range_tree_nodes :
                continue
            range = self.range_tree_nodes[symbol]
            if (range.is_tiling_axis1 or range.is_tiling_axis2) and (symbol not in axis_list):
                line = line.replace(f"{range.name}", f"{range.name}_prime")
                mask = mask.replace(f"{range.name}", f"{range.name}_prime")
                axis_list.append(symbol)
        return line, mask

    # apply xxx_prime var in case dim are permuted
    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        var = self.args.output(name)
        original_index = index
        indexing = self.indexing(index, dense_indexing=True, block_ptr=mode is None)
        index_str = indexing.index_str
        value_str = f"{value}"

        # need to reshape when value's dimensions > 2, e.g. (XBLOCK,1,RBLOCK)
        is_permuted = self.need_permuted(index)

        mask_str = indexing.mask_str
        if is_permuted:
            index_str, mask_str = self.apply_var_prime(index, index_str, indexing.mask_str)
            value_str = value_str.replace(f"{value}", f"{value}.permute(1,0)")

        advance_block_ptr = None
        if isinstance(indexing, BlockPtrOptions):
            block_ptr, advance_block_ptr, other = self.codegen_block_ptr(
                name, var, indexing
            )
            # block_ptr stores don't do implicit casting
            line = self.codegen_block_ptr_store_line(
                name, indexing, block_ptr, value, other
            )
        elif mode is None:
            line = f"tl.store({var} + ({index_str}), {value_str}, {mask_str})"
            if len(self.axis2_list) > 1 :
                line = f"tl.store({var} + ({index_str} + tl.arange(0,1) ), {value_str}, {indexing.mask_str})"

        elif mode == "atomic_add":
            line = f"tl.atomic_add({var} + ({index_str}), {value_str}, {indexing.mask_str})"
        else:
            raise NotImplementedError(f"store mode={mode}")

        self.stores.writeline(DeferredLine(name, line))
        if advance_block_ptr:
            self.stores.writeline(advance_block_ptr)

        if not self.inside_reduction:
            self.outside_loop_vars.add(value)


    @staticmethod
    def _get_next_scheduler_node(node_schedule, current_node):
        found_current = False if current_node else True
        for node in node_schedule :
            if isinstance(node, SchedulerNode) :
                if not found_current and node.get_name() == current_node.get_name() :
                    found_current = True
                    continue
                if found_current  :
                    return node
        return None

    #fixme, this seems not reliable, need to refactor .
    def get_next_scheduler_node(self, node):
        return self._get_next_scheduler_node(self.node_schedule, node)

    def get_prev_scheduler_node(self, node):
        return self._get_next_scheduler_node(reversed(self.node_schedule), node)

    # to generate the shape of the accumulator of RBLOCK loop
    def dense_size_list(self, is_permute) -> List[str]:

        sizes = []
        if self.numof_reduction_axis() > 1:
            sizes = [f"RBLOCK_{axis}" for axis in self.axis2_list]
            return sizes
        if self.persistent_reduction and self.axis2 is None :
            sizes = ["RBLOCK" ]
            return sizes
        # current computedbuffer is reduction
        cb_is_reduction = self.inside_reduction if not self.current_node else isinstance(self.current_node.node.data, ir.Reduction)

        for tree in self.sorted_axis:
            if tree.is_tiling_axis1 :
                sizes.append("XBLOCK_SUB")
            elif tree.is_tiling_axis2:
                sizes.append("RBLOCK")

        if cb_is_reduction and self.inside_reduction and self.is_higher_order_reduction() or is_permute:
            sizes = reversed(sizes)

        return sizes

    def dense_size_str(self, is_permute = False):
        sizes = self.dense_size_list(is_permute)
        if self.numof_reduction_axis() > 1:
            return f"[{'* '.join(sizes)}]"
        return f"[{', '.join(sizes)}]"

    def filter_masks(self, mask_vars):
        for node in self.sorted_axis:
            if not(node.is_tiling_axis1 or node.is_tiling_axis2):
                mask_vars.discard(f"{node.name}_mask")
            if len(self.axis2_list) > 1 and  not node.is_tiling_axis2:
                mask_vars.discard(f"{node.name}_mask")


    # and add to shape to value
    def reduction_resize(self, value):
        ndims = self.triton_tensor_ndim()
        if ndims == 1:
           return f"triton_helpers.promote_to_tensor({value})"
        is_higher_order_reduction = self.is_higher_order_reduction()

        expand_str = "1," if is_higher_order_reduction else ",1"
        if is_higher_order_reduction:
            return f"{value}.reshape({expand_str}XBLOCK_SUB)"
        else:
            return f"{value}.reshape(XBLOCK_SUB{expand_str})"

    def get_axis_direction(self, is_axis1, reversed = False ):

        if not self.inside_reduction :
            if self.numof_tiling_axis() > 1 :
                result = AxisDirection.Vertical if is_axis1 else AxisDirection.Horizontal
            else :
                result = AxisDirection.Flat
        else  :
            if is_axis1 :
                result = AxisDirection.Horizontal if V.kernel.is_higher_order_reduction() else AxisDirection.Vertical
            else :
                result = AxisDirection.Vertical if V.kernel.is_higher_order_reduction() else AxisDirection.Horizontal

        result =  reverse_direction(result) if reversed else result
        return result

    def is_higher_order_reduction(self, check_prev_node = False ):
        if self.numof_reduction_axis() > 1 :
            return False
        assert self.inside_reduction
        if self.inside_high_order_reduction :
            return self.inside_high_order_reduction

        node = self.current_node if self.current_node is not None else self.get_prev_scheduler_node(None)
        if node is None or not isinstance(node, SchedulerNode) :
            return False

        reduction = node.node.data
        while check_prev_node and reduction is not None and  not isinstance(reduction, ir.Reduction) :
            node = self.get_prev_scheduler_node(node)
            if node is None :
                reduction = None
            else :
                reduction = node.node.data


        if reduction is None or not isinstance(reduction, ir.Reduction) :
            return False
        if not hasattr(reduction, "reduced_idx") :
            return False

        reduced_order = reduction.reduced_idx[0]
        is_last_axis = all(_ < reduced_order for _ in reduction.kept_idx)
        self.inside_high_order_reduction = not is_last_axis
        return self.inside_high_order_reduction
    def get_axis_dtype(self, axis):
        dtype = None
        if axis is None :
            return None
        for node in self.node_schedule :
            if node in (EnableReduction, DisableReduction) :
                continue
            if axis.symbol() in node._body.indexing_map :
                dtype = V.graph.get_dtype(node.node.name)
                break
        if dtype is None :
            should_break_all = False
            for node in self.node_schedule:
                if should_break_all:
                    break
                if node in (EnableReduction, DisableReduction):
                    continue
                for key, value in node._body.indexing_map.items():
                    if key  in self.range_tree_nodes :
                        dim = self.range_tree_nodes[key]
                    else :
                        dim = self.range_tree_nodes_removed[key]

                    if dim.parent == axis.parent :
                        dtype = V.graph.get_dtype(node.node.name)
                        should_break_all = True
                        break
        return dtype
    def create_inductor_meta(self):
        mutated_args = set()
        for mutation in self.mutations:
            if mutation in self.args.input_buffers:
                mutated_args.add(self.args.input_buffers[mutation])
            if (
                    mutation in self.args.inplace_buffers
                    and mutation not in V.graph.removed_buffers
                    and mutation not in self.removed_buffers
            ):
                mutated_args.add(self.args.inplace_buffers[mutation].inner_name)
            if mutation in self.args.output_buffers:
                mutated_args.add(self.args.output_buffers[mutation])
        mutated_args = sorted(mutated_args)
        axis1_order = self.range_tree_nodes[self.axis1].sorted_order if self.axis1 is not None else None
        axis2_order = self.range_tree_nodes[self.axis2].sorted_order if self.axis2 is not None else None
        split_axis_dtype = self.get_axis_dtype(self.split_axis)
        inductor_meta = {
            "autotune_hints": set(self.autotune_hints),
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            "mutated_arg_names": mutated_args,
            "no_x_dim": self.no_x_dim,
            # Due to breaking change of triton 3.0, the original invocation is broken
            "backend_hash": self.patch_triton_hash(),  # torch.utils._triton.triton_hash_with_backend(),
            #"high_order_reduction" : self.inside_reduction and self.is_higher_order_reduction(True) ,
            "split_axis_order" : self.split_axis.sorted_order if self.split_axis is not None else None,
            "axis1_order" : axis1_order,
            "axis2_order": axis2_order,
            "low_dims" : self.low_dims,
            "numof_reduction_axis": self.numof_reduction_axis(),
            "split_axis_dtype":split_axis_dtype
        }
        return inductor_meta
    def reduction_dim(self):
        assert self.inside_reduction
        if self.numof_reduction_axis() > 1:
            return 0
        return 0 if self.is_higher_order_reduction() or len(self.sorted_axis) ==1 else 1
    def reduction_var(self):
        var = self.axis2
        return var


    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, Tuple[CSEVariable, ...]],
    ) -> Union[CSEVariable, Tuple[CSEVariable, ...]]:
        assert self.inside_reduction
        masks = {f"{node.symbol()}_mask" for node in self.sorted_axis}
        self.filter_masks(masks)
        masks = sorted(masks)
        if self._load_mask:
            masks.append(self._load_mask)
        reduction_range_prefix = self.range_trees[-1].prefix

        dense_size_str = self.dense_size_str(False)
        value = self._map_tuple_or_scalar(
            lambda v: self.cse.generate(
                self.compute, f"tl.reshape({v}, {dense_size_str})"
            ),
            value,
        )

        dim: int
        root_op: str

        def final_reduction(value):
            #use_helper = reduction_type in {"any", "max", "min", "prod"}
            module = "tl" # use tl
            if reduction_type in {"max", "min"}:
                return self.reduction_resize(f"{module}.{reduction_type}({value}, {dim})" #use tl.max
                )
            return self.reduction_resize(f"{module}.{reduction_type}({value}, {dim})")

        def final_argreduce(buffer, result_var, value, index):
            buffer.splice(
                f"""\
                _, {result_var}_tmp = triton_helpers.{root_op}_with_index({value}, {index}, {dim})
                {result_var} = {self.reduction_resize(f'{result_var}_tmp')}
                """
            )
        def get_reduction_axis() :
            return list(self.range_tree_nodes.values())[-1]

        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]

        dim = self.reduction_dim()
        acc_type = triton_acc_type(src_dtype)
        result_var: Any = self.cse.newvar()
        result_var.mask_vars = {var for var in masks if var[0] != "r"}
        cond = " & ".join(masks)


        def where_cond(tval, fval):
            if not cond:
                return tval
            return TritonKernelOverrides.where(cond, tval, fval)

        if self.persistent_reduction:
            default = ir.Reduction.default_value(reduction_type, src_dtype)
            default = self._map_tuple_or_scalar(triton_constant, default)

            def _mask_value(value, default):
                return self.cse.generate(self.compute, where_cond(value, default))
            # fixme masked_value doesn't work dual reduction
            if self.numof_reduction_axis() == 1 :
                if isinstance(value, tuple):
                    masked_value = [_mask_value(v, d) for v, d in zip(value, default)]
                else:
                    masked_value = _mask_value(value, default)
            else :
                masked_value = value

            if reduction_type in {"argmax", "argmin", "max", "min"}:
                reduce_axis = get_reduction_axis()
                broadcast_string: str
                if self.is_higher_order_reduction():
                    broadcast_string = f"tl.broadcast_to({reduce_axis.symbol()}.reshape({reduction_range_prefix.upper()}BLOCK,1), {masked_value}.shape)"
                else:
                    broadcast_string = f"tl.broadcast_to({reduce_axis.symbol()}.reshape(1,{reduction_range_prefix.upper()}BLOCK), {masked_value}.shape)"
                accumulator_index = str(
                    self.cse.generate(
                        self.compute,
                        broadcast_string
                    )
                )
                if reduction_type == "argmax" or reduction_type == "argmin":
                    root_op = {"argmax": "max", "argmin": "min"}[reduction_type]
                    final_argreduce(
                        self.compute, result_var, masked_value, accumulator_index
                    )
                elif reduction_type == "max" or reduction_type == "min":
                    result_var = self.cse.generate(
                        self.compute, final_reduction(masked_value)
                    )
            elif reduction_type == "welford_reduce":
                assert False, "welford_reduction is not supported now.."
            elif reduction_type == "welford_combine":
                assert False, "welford_combine is not supported now.."
            else:
                result_var = self.cse.generate(
                    self.compute, final_reduction(masked_value)
                )
        else:
            accumulator = f"_{result_var}"
            default = ir.Reduction.default_accumulator(reduction_type, src_dtype)
            default = self._map_tuple_or_scalar(triton_constant, default)
            if not isinstance(default, tuple):
                self.prefix.writeline(
                    f"{accumulator} = tl.full({self.dense_size_str()}, {default}, {acc_type})"
                )

            if reduction_type in {"argmax", "argmin"}:
                accumulator_index = f"_{result_var}_index"
                long_max = torch.iinfo(torch.int64).max
                self.prefix.writeline(
                    f"{accumulator_index} = tl.full({self.dense_size_str()}, {long_max}, tl.int64)"
                )
                root_op = {"argmax": "max", "argmin": "min"}[reduction_type]

                self.compute.splice(
                    f"""\
                {accumulator}_next, {accumulator_index}_next = triton_helpers.{root_op}imum_with_index(
                    {accumulator}, {accumulator_index}, {value}, {reduction_range_prefix}index
                )
                {accumulator} = {where_cond(f'{accumulator}_next', accumulator)}
                {accumulator_index} = {where_cond(f'{accumulator_index}_next', accumulator_index)}
                """
                )
                final_argreduce(self.suffix, result_var, accumulator, accumulator_index)
            elif is_welford_reduction(reduction_type):
                assert False, "welford_reduction is not supported now.."
            else:
                combine_fn = ir.get_reduction_combine_fn(reduction_type, src_dtype)
                updated = combine_fn(accumulator, value)
                self.compute.writeline(
                    f"{accumulator} = {where_cond(updated, accumulator)}"
                )

                if src_dtype == torch.bool:
                    accumulator = f"{accumulator}.to(tl.int8)"
                    result_type = triton_compute_type(dtype)
                    self.suffix.writeline(
                        f"{result_var} = {final_reduction(accumulator)}.to({result_type})"
                    )
                else:
                    self.suffix.writeline(
                        f"{result_var} = {final_reduction(accumulator)}"
                    )

        self.cse.reduction_cache[cache_key] = result_var

        if isinstance(result_var, tuple):
            self.outside_loop_vars |= set(result_var)
        else:
            self.outside_loop_vars.add(result_var)

        return result_var
    #XBLICK:split size, XBLOCK_SUB : tile1 size, RBLOCK:tile2 size
    def add_autotune_args(self, argdefs):
        # no tiling in this case
        if self.persistent_reduction and self.axis2 is None:
            return
        argdefs.append(f"XBLOCK: tl.constexpr")
        if self.numof_reduction_axis() <= 1 :
            argdefs.append(f"XBLOCK_SUB: tl.constexpr")
        if self.axis2 is not None and not self.persistent_reduction:
            argdefs.append(f"RBLOCK: tl.constexpr")

    def _get_heuristic(self):
        if self.persistent_reduction:
            assert self.inside_reduction
            return "persistent_reduction_npu_index"
        elif self.inside_reduction:
            return "reduction_npu_index"
        return "pointwise_npu_index"

    def need_broadcast(self, index: sympy.Expr):
        tiling_axis = [False, False]
        for axis in index.free_symbols:
            if axis not in self.range_tree_nodes :
                continue
            if self.range_tree_nodes[axis].is_tiling_axis1:
                tiling_axis[0] = True
            elif self.range_tree_nodes[axis].is_tiling_axis2:
                tiling_axis[1] = True
        #implict broadcast
        result = (self.numof_tiling_axis() > 1 and not self.persistent_reduction) and (tiling_axis[1] ^  tiling_axis[0])
        result = result and self.find_axis2_in_indexing()
        return  result,  tiling_axis

    def current_node_has_permute(self):
        if not self.current_node :
            return False
        for index in self.current_node._body.indexing.values():
             if self.need_permuted(index) :
                return True
        return False
    def need_permuted(self, index: sympy.Expr):
        if self.numof_tiling_axis() <= 1 :
            return False

        need_permute = False
        tmp_list = []
        coefficients_dict = index.as_coefficients_dict()
        need_permute_axis1 = False
        need_permute_axis2 = False
        for key,value in coefficients_dict.items():
            if not key.free_symbols :
                continue
            key = list(key.free_symbols)[0]
            if key not in self.range_tree_nodes :
                continue
            axis = self.range_tree_nodes[key]
            # normally, axis2 is lowest dimension, except for higher_order_reduction
            if (self.inside_reduction and self.is_higher_order_reduction(True)) :
                if axis.is_tiling_axis1 and value > sympy.Integer(1):
                    need_permute_axis1 = True
            elif axis.is_tiling_axis2 and value > sympy.Integer(1) :
                need_permute_axis2 = True if self.numof_reduction_axis() <= 1 else isinstance(axis.expr, ModularIndexing)
            tmp_list.append(True if value > sympy.Integer(1) else False)

        # If all axes have coefficients greater than 1,
        # then the stride is not 1, and in this case, return false,
        # indicating that the transpose is not required.
        if all(tmp_list):
            return False
        return need_permute_axis1 or need_permute_axis2

    def get_reshape_dense_str(self, tiling_axis):
        # there must be one tiling asis missing
        assert tiling_axis[1] or tiling_axis[0]
        sizes = ["XBLOCK_SUB", "1"]
        if not tiling_axis[0] :
            sizes = ["1", "RBLOCK"]

        if self.inside_reduction and self.is_higher_order_reduction():
            sizes = reversed(sizes)
        return f"[{', '.join(sizes)}]"

    def get_reshape_str(self, tiling_axis, check_prev_node = True):
        # there must be one tiling asis missing
        assert tiling_axis[1] or tiling_axis[0]
        sizes = ["XBLOCK_SUB", "RBLOCK"]
        if not tiling_axis[0] :
            sizes[0] = "1"
        elif not tiling_axis[1] :
            sizes[1] = "1"
        if self.inside_reduction and self.is_higher_order_reduction(check_prev_node):
            sizes = reversed(sizes)

        return f"[{', '.join(sizes)}]"

    def get_broadcast_dense_str(self, tiling_axis, check_prev_node = True):
        # there must be one tiling asis missing
        assert tiling_axis[1] or tiling_axis[0]
        sizes = ["XBLOCK_SUB", "RBLOCK"]
        if self.inside_reduction and self.is_higher_order_reduction(check_prev_node):
            sizes = reversed(sizes)
        #elif not tiling_axis[0] :
        #    sizes = reversed(sizes)
        return f"[{', '.join(sizes)}]"

    #broadcast, permute handling
    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        original_index = index
        is_permuted = self.need_permuted(index)
        store_cache = self.cse.store_cache
        if name in store_cache:
            broadcasted, tiling_axis = self.need_broadcast(original_index)
            result_var = store_cache[name]
            if broadcasted:
                line = f"{result_var}.broadcast_to({self.get_broadcast_dense_str(tiling_axis, True)})"
                buffer = self.compute if self.persistent_reduction else self.loads
                result_var = self.cse.generate(buffer, line)
            elif is_permuted:
                line = f"{result_var}.permute(1,0)"
                buffer = self.compute if self.persistent_reduction else self.loads
                result_var = self.cse.generate(self.loads, line)
            return result_var

        need_broadcast, tiling_axis = self.need_broadcast(index)
        indirect_indexing = self.is_indirect_indexing(index)
        indexing = self.indexing(index, block_ptr=True)
        has_rindex = indexing.has_rindex()
        has_tmpmask = indexing.has_tmpmask()
        is_coalesced = any(
            i == 1 for i in self.get_strides_of_load(original_index).values()
        )
        ep = ""
        if (
                (has_tmpmask or has_rindex)
                and V.graph.get_dtype(name) != torch.bool
                and indexing.has_mask()
        ):
            other = ", other=0.0"
        else:
            other = ""

        advance_block_ptr = None
        append_broadcast = None

        if V.graph.is_unspec_arg(name):
            line = var
        else:
            if isinstance(indexing, BlockPtrOptions):
                block_ptr, advance_block_ptr, other = self.codegen_block_ptr(
                    name, var, indexing, other
                )
                line = f"tl.load({block_ptr}{other}{ep})"
                # add needed size=1 dimensions
                line = triton_reshape(
                    line, indexing.block_shape, indexing.reshape_suffix
                )
            elif isinstance(original_index, sympy.Integer):
                line = f"tl.load({var} + ({original_index}))"
                num_size = len(self.dense_size_list(is_permuted))
                append_broadcast = "[1, 1]" if (num_size > 1) else "[1]"
            else:
                index_str = indexing.index_str
                mask_str = indexing.mask_str
                if is_permuted:
                    index_str, mask_str = self.apply_var_prime(index, index_str, mask_str)
                line = f"tl.load({var} + ({index_str}), {mask_str}{ep}{other})"

            dtype = V.graph.get_dtype(name)
            if dtype in (torch.float16, torch.bfloat16):
                line += ".to(tl.float32)"
            if dtype == torch.bool and torch.version.hip is None:
                line += ".to(tl.int1)"
        if has_tmpmask:
            # Masked loads must come after the mask is computed
            load_buffer = self.compute
        elif (
            self.inside_reduction
            and self.range_trees[-1].is_loop
            and not indirect_indexing
            and not has_rindex
        ):
            # can lift a common load outside of reduction loop
            # One exception is when this is an indirect_load.
            load_buffer = self.prefix

        else:
            load_buffer = self.loads
        result_var = self.cse.generate(load_buffer, line)
        assert isinstance(result_var, TritonCSEVariable)
        result_var.mask_vars = indexing.mask_vars  # type: ignore[assignment]

        if append_broadcast and append_broadcast != '[]':
            line = f"tl.broadcast_to({result_var}, {append_broadcast})"
            result_var = self.cse.generate(load_buffer, line)
        elif need_broadcast and not indirect_indexing:
            #reshape_str = self.get_reshape_str(tiling_axis)
            #.reshape({reshape_str})
            line = f"{result_var}.broadcast_to({self.get_broadcast_dense_str(tiling_axis)})"
            result_var = self.cse.generate(load_buffer, line)
        elif is_permuted:
            line = f"{result_var}.permute(1,0)"
            result_var = self.cse.generate(self.loads, line)

        if advance_block_ptr:
            load_buffer.writeline(advance_block_ptr)

        if not self.inside_reduction or (not indexing.has_rmask() and not has_rindex):
            self.outside_loop_vars.add(result_var)

        return result_var

    #1. only remove the line which asserts index var should be in "xyr"
    #2. don't do simplify_indexing, which combine continuous dims
    #3. removed block_ptr, removed dense mask/broadcast support
    # fixme, dense_mask_vars should be generated from sorted_axis
    def indexing(
            self,
            index: sympy.Expr,
            *,
            copy_shape=None,
            dense_indexing=False,
            override_mask=None,
            block_ptr=False,
    ) -> Union[IndexingOptions, BlockPtrOptions]:
        """
        Compute the index and mask to pass to tl.load() or tl.store()
        """
        #index = self.simplify_indexing(index)
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
        # if simple replacements didn't get rid of floor/ceil, try full subs
        if len(index.atoms(sympy.floor)) or len(index.atoms(sympy.ceiling)):
            index = index.subs(V.graph.sizevars.precomputed_replacements)
        if len(index.atoms(sympy.ceiling)):
            for a in index.atoms(sympy.ceiling):
                # for nested exprs, atoms yields top level first (?)
                # so if everything goes fine, lower level replacements will come up empty
                symbols = a.free_symbols
                if len(symbols) > 0 and all(
                        s.name.startswith("s") or s.name.startswith("ps") for s in symbols
                ):
                    replacements = {a: V.graph.sizevars.lookup_precomputed_size(a)}
                    index = sympy_subs(index, replacements)

        #index = self.simplify_indexing(index)
        index_vars = index.free_symbols
        has_rindex = False

        mask_vars: Set[str] = set()
        for var in index_vars:
            assert isinstance(var, sympy.Symbol)
            has_rindex = has_rindex or var.name.startswith("r")
            if override_mask:
                pass
            elif var.name.startswith("tmp"):
                # indirect indexing
                cse_var = self.cse.varname_map[var.name]
                mask_vars.update(cse_var.mask_vars)
            elif var.name.startswith(("s", "ps", "i")):
                pass
            else:
                # var is one of xN, yN or rN
                # assert var.name[0] in "xyr", var.name
                mask_vars.add(f"{var.name}_mask")

        expand_str = None
        index_str = self.index_to_str(index)
        is_permute = self.need_permuted(index)
        if isinstance(index, sympy.Integer):
            expand_str = f"{copy_shape}.shape" if copy_shape else self.dense_size_str(is_permute)
            if (index != 0):
                index_str = f"tl.full({expand_str}, {index_str}, tl.int32)"
            else:
                index_str = f"tl.arange(0,1)"
            return IndexingOptions(index_str, set(), "None", expand_str, has_rindex)

        if override_mask:
            mask_vars = {override_mask}
        if self._load_mask:
            mask_vars.add(self._load_mask)
        self.filter_masks(mask_vars)
        mask_str = " & ".join(sorted(map(str, mask_vars))) if mask_vars else "None"
        return IndexingOptions(index_str, mask_vars, mask_str, expand_str, has_rindex)  # type: ignore[arg-type]


    #support split multiple ranges (instead of double) from one flatten range, triple-ranges are needed in mamba model
    @staticmethod
    def _split_iteration_ranges(
        groups: Iterable[sympy.Expr], lengths: List[List[sympy.Expr]]
    ):
        sv = V.graph.sizevars
        new_ranges: List[List[sympy.Expr]] = [[] for _ in groups]
        remaining = [sv.simplify(g) for g in groups]
        for i, group in enumerate(remaining) :
            if isinstance(group, (list, tuple)):
                remaining[i] = NumelList(group).numels()

        var_count = itertools.count()

        def add_range(i, expr):
            expr = sv.simplify(expr)
            if not sv.statically_known_multiple_of(remaining[i], expr):
                raise CantSplit()
            # guard on the last item out
            remaining[i] = FloorDiv(remaining[i], expr)
            new_ranges[i].append(expr)
            return next(var_count)

        def make_combined(strides, index_list):
            def getter(flat_vars):
                expr = sympy.Integer(0)
                for stride, index in zip(strides, index_list) :
                    expr = stride * flat_vars[index] + expr
                return expr

            return getter

        def size_hints(group):
            if isinstance(group, (list,tuple)) :
                return sv.size_hint(NumelList(group).numels())
            return sv.size_hint(group)
        def add_multiple_range(size, return_getters):
            # need to break size in multiple
            index_list = []
            stride_list = []
            group = current_group
            remained_size = size
            while group < len(remaining)  and remaining[group] > 1 :
                group_size = remaining[group]
                # size should be divisible by group_size
                if not sv.statically_known_multiple_of( remained_size, group_size ):
                    raise CantSplit()
                index_list.append(add_range(group, group_size))
                remained_size = FloorDiv(remained_size, group_size)
                stride_list.append(remained_size)
                group = group + 1
            if remained_size != 1 :
                raise CantSplit()
            return_getters.append(make_combined(stride_list, index_list))

        return_getters_groups = []
        current_group = 0

        for length_group in lengths:
            return_getters = []
            for size in length_group:
                if sv.statically_known_equals(size, 1):  # type: ignore[arg-type]
                    return_getters.append(lambda _: sympy.Integer(0))
                    continue

                while (
                    current_group < len(remaining)
                    and size_hints(remaining[current_group]) == 1
                ):
                    # scroll to next group with remaining elements
                    current_group += 1

                if sv.size_hint(size) > size_hints(remaining[current_group]):
                    #add multiple ranges (two or more) to the list, as well as the getter funcs
                    add_multiple_range(size, return_getters)
                else:
                    return_getters.append(
                        operator.itemgetter(add_range(current_group, size))
                    )
            return_getters_groups.append(return_getters)

        assert all(
            V.graph.sizevars.size_hint(s) == 1 for s in remaining
        ), f"failed to set ranges {remaining} {lengths}"

        return new_ranges, return_getters_groups

    # just to override load method of CSEProxy, however, CSEProxy is an inner which can not be monkey patched,
    # we need to override the whole inner class
    def __enter__(self):
        # TODO: hoist this to top level
        class CSEProxy:
            self.name = "CSEProxy"

            @staticmethod
            def __getattr__(name: str) -> Callable[..., CSEVariable]:  # type: ignore[misc]
                def inner(*args, **kwargs):
                    # TritonTemplateKernel has no current_node
                    buf_bounds = ValueRanges.unknown()
                    if hasattr(V.interpreter, "current_node"):
                        fx_node = V.interpreter.current_node
                        assert isinstance(self.node_to_bounds, dict)
                        buf_bounds = self.node_to_bounds.get(
                            fx_node, ValueRanges.unknown()
                        )

                    value = getattr(parent_handler, name)(*args, **kwargs)  # type: ignore[has-type]

                    def do_cse(v):
                        csevar = self.cse.generate(self.compute, v, bounds=buf_bounds)
                        csevar.update_on_args(name, args, kwargs)
                        return csevar

                    return pytree.tree_map(do_cse, value)

                return inner

            @staticmethod
            def indirect_indexing(
                var: CSEVariable, size: sympy.Expr, check: bool = True
            ):
                # Skip CSE since this doesn't return an expression

                if var.bounds.lower < 0:  # type: ignore[operator]
                    new_bounds = ValueRanges.unknown()
                    if var.bounds != ValueRanges.unknown() and isinstance(
                        size, sympy.Number
                    ):
                        # Take the negative part of the bound and add size to it
                        # Then take union of that and the positive part
                        # This is a tighter bound than that of a generic ops.where, as we have info on the cond
                        neg = var.bounds & ValueRanges(-sympy.oo, -1)
                        new_bounds = ValueRanges(neg.lower + size, neg.upper + size)
                        # We don't have a good way of representing the empty range
                        if var.bounds.upper >= 0:  # type: ignore[operator]
                            pos = var.bounds & ValueRanges(0, sympy.oo)
                            new_bounds = new_bounds | pos

                    stm = ops.add(var, self.rename_indexing(size))
                    # Mixed negative and non-negative
                    if var.bounds.upper >= 0:  # type: ignore[operator]
                        lt = ops.lt(var, "0")
                        stm = ops.where(lt, stm, var)
                    new_var = self.cse.generate(self.compute, stm, bounds=new_bounds)

                    new_var.update_on_args("index_wrap", (var,), {})
                    var = new_var

                if self.generate_assert(check):
                    mask = self.load_mask(var)

                    # An assertion line may have been written already, if so just
                    # update the max size.
                    map_key = (var, mask)
                    existing_size, _ = self.indirect_max_sizes.get(
                        map_key, (None, None)
                    )
                    if existing_size is not None:
                        size = sympy.Min(size, existing_size)
                    else:
                        pass
                    self.indirect_max_sizes[map_key] = (size, self.index_to_str(size))
                return sympy_index_symbol(str(var))

            @staticmethod
            def load(name: str, index: sympy.Expr) -> CSEVariable:
                if name in self.cse.invalidated_stores:
                    V.kernel.must_keep_buffers.add(name)
                if free_symbol_startswith(index, "tmp"):
                    return self.indirect_load(name, index)
                store_cache = self.cse.store_cache
                if name in store_cache:
                    return self.load(name, index)
                #    return store_cache[name]
                return self.load(name, index)

            @staticmethod
            def store(
                name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
            ) -> None:
                self.store_buffer_names.add(name)
                if mode is None:
                    self.cse.store_cache[name] = value
                    if self.current_node:
                        for other_name in self.current_node.get_mutations():
                            self.cse.store_cache[other_name] = value
                if name not in V.graph.removed_buffers:
                    return self.store(name, index, value, mode=mode)
                else:
                    return None  # type: ignore[return-value]

            @staticmethod
            def store_reduction(name: str, index: sympy.Expr, value: CSEVariable):
                self.store_buffer_names.add(name)
                self.cse.store_cache[name] = value
                if self.current_node:
                    for other_name in self.current_node.get_mutations():
                        self.cse.store_cache[other_name] = value

                if name not in V.graph.removed_buffers:
                    return self.store_reduction(name, index, value)

            @staticmethod
            def reduction(
                dtype: torch.dtype,
                src_dtype: torch.dtype,
                reduction_type: ReductionType,
                value: Union[CSEVariable, Tuple[CSEVariable, ...]],
            ) -> Union[CSEVariable, Tuple[CSEVariable, ...]]:
                return self.reduction(dtype, src_dtype, reduction_type, value)

            @staticmethod
            def scan(
                dtype: torch.dtype,
                combine_fn: Callable[[CSEVariable, CSEVariable], CSEVariable],
                value: CSEVariable,
                init: int,
            ) -> CSEVariable:
                return self.scan(dtype, combine_fn, value, init)

            @staticmethod
            def bucketize(
                values: CSEVariable,
                offsets_name: str,
                offsets_size: sympy.Expr,
                indexing_dtype: torch.dtype,
                right: bool,
            ) -> CSEVariable:
                return self.bucketize(
                    values, offsets_name, offsets_size, indexing_dtype, right
                )

        # Use sympy to check protocol implemented correctly
        def _typecheck_CSEProxy(h: CSEProxy) -> OpsHandler[CSEVariable]:
            return h

        super().__enter__()
        assert self.overrides
        parent_handler = self.overrides(V.get_ops_handler())
        self.exit_stack.enter_context(V.set_ops_handler(CSEProxy()))
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self

