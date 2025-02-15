import pdb

from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.utils import ModularIndexing,sympy_subs
import sympy as sympy
from ..config import num_vector_core
import logging
log = logging.getLogger(__name__)
log.level = logging.DEBUG
from torch._inductor.virtualized import V
from torch._inductor.codegen.triton import (  EnableReduction, DisableReduction)
from torch._inductor.utils import  next_power_of_2
from .triton_utils import get_aligned_numel

# split and tiling axis selector
class SplitTiling :
    def __init__(self, kernel : TritonKernel) :
        self.kernel = kernel
        self.indexing = []
        def key(x) :
            # to be higher than x and y
            if x.name[0] == 'w' or x.name[0] == 'v' or x.name[0] == 'p' or x.name[0] == 't':
                return "z" + x.name
            # to be lower than floor_dir
            elif isinstance(x.expr, ModularIndexing):
                return x.name[0] + "0" + x.name[1:]
            else :
                return x.name
            
        kernel.sorted_axis = [x for x in kernel.range_tree_nodes.values()]
        kernel.sorted_axis.sort(reverse=True, key=key)
        for i, dim in enumerate(kernel.sorted_axis):
            dim.sorted_order = i
        
        self.find_lowest_dimension()
        self.should_outer_reduce = False


    # Split 原则1 ：先做维度合并，再切分 。通过维度合并降维降低，split和tiling轴选择策略的复杂性 。
    # Split 原则2: 切分的数量要和AIcore的数量对齐（相同或是倍数）。每个核要分配的split的量一致。每个split形状要一致（包括维度和尺寸）。
    # Split  原则3: 对于规约类融合算子, 从非规约选择切分轴。对于非规约类融合算子, 从所有轴中选切分轴。
    # 为了tiling时刻的低维tilesize最大化，切分轴最好不是低维轴且长度大于aicore的数量 。
    # Split 原则4: 如果高维规约类融合算子，而且高维尺寸非常大（ >= 64KB），低维度尺寸比较小（ <= 32B）, 可以选择对规约轴切分，然后在核间用atomic
    # 原语做规约。
    # Split  原则5 ：根据算子逻辑，优先选择一维发射。
    def select_split_axis(self):
        def select_longest_dim(can_be_low_dim = True):
            longest = -1
            longest_dim = None
            for x in candidates:
                if SplitTiling.great_than(x.length,longest) and (can_be_low_dim or not self.is_lowest_dimension(x)):
                    longest_dim = x
                    longest = x.length
            return longest_dim
        # point-wise : all dims , reduction: outer_reduction dim or non-reduction dims
        is_reduction = lambda x : x.prefix == 'r'
        candidates = [x for x in self.kernel.sorted_axis if not is_reduction(x) or self.should_outer_reduce_me(x) ]
        if self.should_outer_reduce :
            return self.kernel.split_axis

        #longest and not low dims
        longest_dim = select_longest_dim( can_be_low_dim = False )

        # longest and can be low dims
        if longest_dim is None or SplitTiling.less_than(longest_dim.length , int(num_vector_core * 0.8)):
            longest_dim = select_longest_dim( can_be_low_dim = True )
        if longest_dim is not None :
            self.kernel.split_axis = longest_dim
            self.kernel.split_axis.is_split_axis = True
        elif len(self.kernel.sorted_axis) > 0:
            longest_dim = self.kernel.sorted_axis[0]
            self.kernel.split_axis = longest_dim
            self.kernel.split_axis.is_split_axis = True
        
        return longest_dim

    # Tiling 原则1：切分要照顾所有load / store 中索引表达式的中的低维轴 ：所有的低维轴都被切分 从而成为tiling 轴。写代码的时候对所有的tiling
    # 轴通过make_range产生连续索引，从而保证load / store的连续性。
    # Tiling 原则2 ：规约的tile必须要二维。 对于低维规约算子，规约轴和至少一个非规约轴要选择为tiling轴。对于高维规约，规约轴和低维轴要选择为tiling轴
    #    对于是多维规约, 所有的规约轴都要选择为tiling 轴 。
    # Tiling 原则3: 如果tiling轴是低维，在该轴上的切分的尺寸要与SIMD的BlockSize 对齐（32bytes）
    # Tiling 原则4: 低维轴的tile size 越大，性能越好。这个其实autotune 的原则，放在这里只是为了更好解释用例中使用的数值 。

    # fixme, two tiling axis might be insufficient when there're 3 or more low-dims in indexing
    def select_tiling_axis(self ):
        # True :self.kernel.axis2 is Not None and all reduction axis selected, False : other cases
        def axis2_selection_done(axis) :
            if self.kernel.total_numels <= 1  :
                return True
            elif self.kernel.axis2 is not None :
                is_reduction = axis.prefix == "r"
                if not is_reduction :
                    return True
                reduction_axis = self.kernel.numof_reduction_axis()
                return True if reduction_axis <= 1 else len(self.kernel.axis2_list) == reduction_axis
            else :
                return False
    
        if self.kernel.axis2 is not None or self.kernel.axis1 is not None:
            return
        # two or more reduction axises, need to flatten reduction dims to one to do 1 dim reduction .
        if self.kernel.numof_reduction_axis() > 1:
            self.kernel.persistent_reduction = True
        biggest = -1
        dims = self.kernel.sorted_axis
        if self.kernel.split_axis is None :
            self.select_split_axis()
        
        if self.kernel.split_axis is None :
            return
        # select tiling_axis2 then tiling_axis1, for reduction, all reduction axis will be selected as tiling_axis2
        for i in range(len(dims)-1, -1, -1) :
            axis = dims[i]
            numel = axis.length
            if isinstance(numel, (sympy.Symbol, sympy.Expr)) and  not isinstance(numel, sympy.Integer) :
                  numel = numel.subs(V.graph.sizevars.var_to_val)
            if axis.is_split_axis :
                dtype = self.kernel.get_axis_dtype(axis)
                _, numel = SplitTiling.decide_nblocks_xblock(numel, len(self.kernel.sorted_axis) <=1, dtype)
            
            # choose reduction axis or low-dim as axis2
            if not axis2_selection_done(axis):
                axis.is_tiling_axis2 =True if SplitTiling.great_than(numel,1) else False
                # axis2 must be the reduction axis in case inside_reduction
                if axis.prefix == "r" :
                    axis.is_tiling_axis2 =True
                if axis.is_tiling_axis2 and  self.kernel.axis2 is None :
                    self.kernel.axis2 = axis.symbol()
                if self.kernel.numof_reduction_axis() > 1 :
                    self.kernel.axis2_list.append(axis.symbol())
                    self.kernel.axis2 = axis.symbol() if isinstance(axis.expr, ModularIndexing) else self.kernel.axis2
            else :
                # for _higher_order_reduction, axis1 must be  the lowest dimension
                if self.kernel.inside_reduction and self.kernel.is_higher_order_reduction() :
                    self.kernel.axis1 = axis.symbol()
                    break

                # low-dim should be selected as another tiling axis
                if self.is_lowest_dimension(axis) :
                    self.kernel.axis1 = axis.symbol()
                    break
                # select the longest in other cases
                if numel > biggest :
                    self.kernel.axis1 = axis.symbol()
                    biggest = numel
                
        if self.kernel.axis1 is not None :
            axis = self.kernel.range_tree_nodes[self.kernel.axis1 ]
            axis.is_tiling_axis1 = True


        log.warning(f"split_tiling numels:{self.kernel.numels} split_axis: {self.kernel.split_axis.symbol()} "
                    f"axis1:{self.kernel.axis1} axis2:{self.kernel.axis2} low_dims:{self.kernel.low_dims}, "
                    f"indexing: {self.indexing}" )



    # fixme the below logic doesn't work when there're two reduction axis, but only one need outer reduction
    def should_outer_reduce_me(self, x):
        should_outer = self.kernel.is_higher_order_reduction(True) and SplitTiling.great_than(x.length, 32768 ) and x.is_loop
        if should_outer :
            self.should_outer_reduce = True
            self.kernel.split_axis = x
            self.kernel.split_axis.is_split_axis = True
        return should_outer
    
    @staticmethod
    def decide_nblocks_xblock(numel, no_axis2, dtype,  xblock = None):
        #no_axis2 mean there's only on dims
        min_aligned_numel = get_aligned_numel(dtype)
        min_xblock = min_aligned_numel if no_axis2 else 1

        # need to keep linearity for low_dims
        if xblock is None :
            xblock = ( numel + num_vector_core -1 ) // num_vector_core if numel > num_vector_core else min_xblock
        
        # fixme, aligning is wasting cores .
        #if (not no_axis2 and  is_low_dim) or same_axis1 :
        xblock = next_power_of_2(xblock)

        nblocks = (numel + xblock -1 ) // xblock
        return nblocks, xblock
    
    # return True when x is the low-dim in indexing
    def is_lowest_dimension(self, x):
        return x.sorted_order in self.kernel.low_dims

    def find_lowest_dimension(self):
        def construct_low_dim() :
            for index in self.indexing:
                coefficients_dict = index.as_coefficients_dict()
                for key, value in coefficients_dict.items():
                    if not key.free_symbols:
                        continue
                    key = list(key.free_symbols)[0]
                    if key not in self.kernel.range_tree_nodes:
                        continue

                    if value == sympy.Integer(1):
                        axis = self.kernel.range_tree_nodes[key]
                        self.kernel.low_dims.add(axis.sorted_order)
        
        # all read index should be considered
        buf_names = [node.node.name for node in self.kernel.node_schedule if
                     node not in (EnableReduction, DisableReduction)]
        for node in self.kernel.node_schedule:
            if node in (EnableReduction, DisableReduction):
                continue
            names = []

            for read in node._body.reads:
                name = node._body.indexing_exprs_name[read]
                read_is_inptr = True
                for arg, expr in node._body.reads_name2expr.items():
                    # read inner buf should be excluded (tmp will cse replace load)
                    if read == expr and (arg[:3] != 'arg' and arg in buf_names):
                        read_is_inptr = False
                if read_is_inptr:
                    names.append(name)
            for key, index in node._body.indexing.items():
                if key in names and index not in self.indexing:
                    self.indexing.append(index)
        if self.kernel.inside_reduction :
            construct_low_dim()
            return
        # for non-reduction, write index should be considered
        for node in self.kernel.node_schedule:
            if node in (EnableReduction, DisableReduction):
                continue
            names = []

            for write in node._body.writes:
                name = node._body.indexing_exprs_name[write]
                names.append(name)
            for key, index in node._body.indexing.items():
                if key in names and index not in self.indexing:
                    self.indexing.append(index)
        
        construct_low_dim()

    @staticmethod
    def convert(x, y):
        xnumel = x
        ynumel = y
        if isinstance(xnumel, (sympy.Symbol, sympy.Expr)) and not isinstance(xnumel, sympy.Integer):
            xnumel = xnumel.subs(V.graph.sizevars.var_to_val)

        if isinstance(ynumel, (sympy.Symbol, sympy.Expr)) and not isinstance(ynumel, sympy.Integer):
            ynumel = ynumel.subs(V.graph.sizevars.var_to_val)

        if isinstance(xnumel, sympy.Integer) and  isinstance(ynumel, int):
            ynumel = sympy.Integer(ynumel)
        
        if isinstance(ynumel, sympy.Integer) and   isinstance(xnumel, int):
            xnumel = sympy.Integer(xnumel)

        return (xnumel, ynumel)
    

    @staticmethod
    def less_than(x, y):
        xnumel, ynumel = SplitTiling.convert(x, y)
        return xnumel < ynumel
    
    @staticmethod
    def great_than(x, y):
        xnumel, ynumel = SplitTiling.convert(x, y)
        return xnumel > ynumel


