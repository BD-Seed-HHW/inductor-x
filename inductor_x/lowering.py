import sympy
from torch._inductor.ir import Reduction
from torch._inductor.utils import sympy_product
from torch._inductor import ir
from torch._inductor.ir import ExpandView, TensorBox, ops_wrapper
from torch._inductor import lowering
from torch._prims_common import (
    is_boolean_dtype,
    is_integer_dtype,
    get_computation_dtype,
)
from torch._inductor.decomposition import decompositions, pw_cast_for_opmath
import torch._ops


def make_reduction(reduction_type: str, override_return_dtype=None):
    def inner(x, axis=None, keepdims=False, *, dtype=None):
        kwargs = _make_reduction_inner(
            x,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            override_return_dtype=override_return_dtype,
        )
        result = Reduction.create(reduction_type=reduction_type, input_node=x, **kwargs)
        if isinstance(
                result.data.data, Reduction
        ):  #Only realize if reduction isn't unrolled
            size = x.get_size()
            axis = set(_validate_reduction_axis(x, axis))
            kept_idx = []
            reduced_idx = []
            for i in range(len(size)):
                if i in axis:
                    reduced_idx.append(i)
                else:
                    kept_idx.append(i)

            setattr(result.data.data, "kept_idx", kept_idx)
            setattr(result.data.data, "reduced_idx", reduced_idx)

            result.realize()
        return result

    return inner


lowering.make_reduction = make_reduction

from torch._inductor.lowering import (
    lowerings,
    make_fallback,
    register_lowering,
    to_dtype,
    # make_reduction,
    # reduce_amax,
    # reduce_amin,
    fallback_cumsum,
    _validate_reduction_axis,
    div,
    squeeze,
    square,
    sub,
    fallback_handler,
    is_boolean_type,
    logical_and,
    make_pointwise,
    _make_reduction_inner,
    _validate_reduction_axis,
)

aten = torch.ops.aten
tr_c10d = torch.ops.tr_c10d
prims = torch.ops.prims

import torch_npu
from torch_npu import npu_dtype_cast


def _init_set(input_list, output_set):
    for fn in input_list:
        output_set.add(fn)
        if isinstance(fn, torch._ops.OpOverloadPacket):
            for overload in fn.overloads():
                other_fn = getattr(fn, overload)
                output_set.add(other_fn)


GENERATE_LIST = [
    aten.mul,
    aten.add,
    aten.sub,
    aten.div,
    aten.exp,
    aten.maximum,
    aten.sum,
    aten.select,
    aten.unsqueeze,
    aten.repeat,
    aten.clone,
    aten.reshape,
    aten.where,
    aten.lt,
    aten.minimum,
    aten.gt,
    aten.le,
    aten.ceil,
    aten.floor,
    aten.rsqrt,
    aten.abs,
    aten.log,
    aten.bitwise_xor,
    aten.amax,
    # backward
    prims.convert_element_type,
    aten.min,
    aten.max,
    aten.erf,
    aten.argmax,
    aten.argmin,
    aten.clamp_min,
    aten.slice,
    aten.neg,
    aten.cat,
    aten.arange,
    aten.expand,
    aten.eq,
    aten.where,
    aten.scalar_tensor,
    aten.ge,
    aten.permute,
    aten.sqrt,
    aten.relu,
    aten.clamp,
    aten.clamp_max,
    aten.mean,
    # npu.npu_dtype_cast
    npu_dtype_cast,
    aten.select_scatter,
    aten.slice_scatter,
    prims.broadcast_in_dim,
    prims.maximum,
    aten.ne
]

GENERATE_LIST2 = [
    "foreach"
]

FALLBACK_LIST = []

# 先删除从lowering已经注册的op，再更新，不然会lowering的时候找到在torch注册的op
LOWERING_OVERLOAD_OP = [
    aten.sum,
    aten.cumsum,
    aten.mean,
    # aten.max,
    # aten.min,
    aten.mul,
    aten.var_mean,
    aten.var,

    # todo: work round for electraModel
    aten.embedding,
    aten.split,
    aten.split_with_sizes,
    aten.nll_loss_forward,
    aten.gather,
    aten.cat,
]


def _register_npu_inductor_fallbacks():
    gen_set = set()
    _init_set(GENERATE_LIST, gen_set)
    overload_op_set = set()
    _init_set(LOWERING_OVERLOAD_OP, overload_op_set)

    for fn in GENERATE_LIST:
        gen_set.add(fn)
        if isinstance(fn, torch._ops.OpOverloadPacket):
            for overload in fn.overloads():
                other_fn = getattr(fn, overload)
                gen_set.add(other_fn)
    # 把不在白名单的op fallback
    for op in lowerings:
        if op not in decompositions and op not in gen_set:
            if isinstance(op, torch._ops.OpOverloadPacket) or \
                    isinstance(op, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
                flag = False
                for gens in GENERATE_LIST2:
                    if str(op).find(gens) != -1:
                        flag = True
                if flag:
                    continue
                else:
                    make_fallback(op)
                    FALLBACK_LIST.append(op)
    # 把需要overload的op在lowering里删除
    for op in overload_op_set:
        if op in lowerings:
            del lowerings[op]

    @register_lowering([aten.sum, prims.sum])
    def sum_(x, axis=None, keepdims=False, *, dtype=None):
        if (
                is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
        ) and dtype is None:
            dtype = torch.int64
        # todo:work for electraModel sum high order reduce,runtime error
        if (axis == [0] and x.get_size() == [32768, 2]):
            return fallback_handler(aten.sum.dim_IntList)(x, dim=axis[0], keepdim=keepdims, dtype=dtype)
        dtype = torch.float32
        fn = make_reduction("sum", override_return_dtype=dtype)
        return fn(x, axis, keepdims, dtype=torch.float32)

    @register_lowering(aten.mean)
    def mean(x, axis=None, keepdim=False, *, dtype=None):
        size = x.get_size()
        if dtype is not None:
            x = to_dtype(x, dtype)
            size = x.get_size()
        axis = _validate_reduction_axis(x, axis)

        # compute in higher-precision until end of mean lowering
        output_dtype = x.get_dtype()
        if output_dtype in (torch.float16, torch.bfloat16):
            x = to_dtype(x, torch.float)
        sum_result = sum_(x, axis, keepdim)
        denom = sympy_product(size[i] for i in axis)
        denom = ir.IndexingConstant(denom, x.get_dtype(), x.get_device())
        denom = ExpandView.create(denom, list(sum_result.get_size()))
        return to_dtype(div(sum_result, denom), output_dtype)

    @register_lowering(aten.cumsum)
    def cumsum(x, axis=None, dtype=None):
        if (
                is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
        ) and dtype is None:
            dtype = torch.int32 # torch.int64->torch.int32
        if len(x.get_size()) == 0:
            assert axis in [0, -1]
            dtype = dtype or x.get_dtype()
            return to_dtype(x, dtype, copy=True)
        return fallback_cumsum(x, dim=axis, dtype=dtype)

    @register_lowering(npu_dtype_cast, type_promotion_kind=None)
    def _convert_npu_type(x: TensorBox, dtype: torch.dtype):
        return to_dtype(x, dtype, copy=True)

    @register_lowering([aten.mul], broadcast=True)
    def mul(a, b):
        if (a.get_size() == [] or a.get_device() == "cpu"):
            return fallback_handler(aten.mul.Scalar)(a, b)

        size = a.get_size()
        numel = 1
        for dim in size:
            numel *= dim
        if numel < 8:
            return fallback_handler(aten.mul.Tensor)(a, b)

        both_bool = is_boolean_type(a) and is_boolean_type(b)
        if both_bool:
            return logical_and(a, b)
        else:
            fn = ops_wrapper(aten.mul.__name__)
            return make_pointwise(fn)(a, b)

    def var_mean_sum_(x, axis, correction, keepdim, return_mean):
        if correction is None:
            correction = 1

        size = x.get_size()
        axis = _validate_reduction_axis(x, axis)
        x_mean = mean(x, axis, keepdim=True)
        if return_mean:
            x_mean.realize()

        diffs = square(sub(x, x_mean))
        sum_result = sum_(diffs, axis, keepdim)

        denom = sympy_product(size[i] for i in axis)
        if correction:
            denom = sympy.Max(denom - correction, 0)
        denom = ir.IndexingConstant(denom, x.get_dtype(), x.get_device())
        denom = ExpandView.create(denom, list(sum_result.get_size()))
        x_var = div(sum_result, denom)
        if not return_mean:
            return (x_var,)

        x_mean = x_mean if keepdim else squeeze(x_mean, axis)
        return x_var, x_mean

    def var_mean_helper_(x, *, axis, correction, keepdim, return_mean):
        out_dtype = x.get_dtype()
        compute_dtype = get_computation_dtype(out_dtype)
        x = to_dtype(x, compute_dtype, copy=False)
        kwargs = dict(
            x=x,
            axis=axis,
            correction=correction,
            keepdim=keepdim,
            return_mean=return_mean,
        )
        output = (
            var_mean_sum_(**kwargs)
            #todo: The welford reduction branch is annotated
            # if use_two_step_variance(x,axis=axis,keepdim=keepdim)
            # else var_mean_welford_(**kwargs)
        )
        output = tuple(to_dtype(x, out_dtype, copy=False) for x in output)
        return output[0] if not return_mean else output

    @register_lowering(aten.var_mean)
    def var_mean(x, axis=None, *, correction=None, keepdim=False):
        return var_mean_helper_(
            x, axis=axis, correction=correction, keepdim=keepdim, return_mean=True
        )

    @register_lowering([aten.var, prims.var])
    def var_(x, axis=None, *, correction=None, keepdim=False):
        return var_mean_helper_(
            x, axis=axis, correction=correction, keepdim=keepdim, return_mean=False
        )

    @register_lowering(aten.embedding, type_promotion_kind=None)
    def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
        return fallback_handler(aten.embedding.default)(weight, indices, padding_idx=-1, scale_grad_by_freq=False,
                                                        sparse=False)

    @register_lowering(aten.cat)
    def cat(inputs, dim=0):
        # todo:work round for electraModel backward
        return fallback_handler(aten.cat.default)(inputs, dim)

    make_fallback(aten._log_softmax)
    make_fallback(aten.gather)
    make_fallback(aten.nll_loss_forward)
