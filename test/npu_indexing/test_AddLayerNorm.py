import time

import torch
from torch import nn
import torch_npu
import triton
import triton.language as tl
from triton.compiler import ASTSource, AttrsDescriptor
import pdb

Z = 64
X = 512
Y = 256

@triton.jit
def test_addLayerNorm_loop(in_ptr0, in_ptr1, out_ptr0,
          XBLOCK : tl.constexpr, XBLOCK_SUB : tl.constexpr, RBLOCK:tl.constexpr,
          Z: tl.constexpr, X: tl.constexpr, R: tl.constexpr,
          Z_STRIDE: tl.constexpr, X_STRIDE: tl.constexpr, R_STRIDE: tl.constexpr,
          ):
    offset: tl.constexpr = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    base2 = tl.arange(0, RBLOCK)
    nsub: tl.constexpr = XBLOCK // XBLOCK_SUB
    loops1: tl.constexpr = nsub
    loops2: tl.constexpr = R // RBLOCK

    for z in range(Z):
        for loop1 in range(loops1):
            off1 = loop1
            x = offset + (off1 * XBLOCK_SUB) + base1[:, None]
            _tmp6 = tl.full([XBLOCK_SUB, RBLOCK], 0, tl.float32)
            _tmpVec0 = tl.full([1, RBLOCK], 0, tl.float32)
            for loop2 in range(loops2):
                r = loop2 * RBLOCK + base2[None, :]
                tmp0 = tl.load(in_ptr0 + (R_STRIDE * r + (X_STRIDE * x) + (Z_STRIDE * z)), None)
                tmp1 = tl.load(in_ptr1 + (R_STRIDE * r + (X_STRIDE * x) + (Z_STRIDE * z)), None)
                tmp2 = tmp0 + tmp1
                _tmp6 = _tmp6 + tmp2

            tmp6 = tl.sum(_tmp6, 1) / R
            tmpMean = tmp6.reshape(XBLOCK_SUB, 1).broadcast_to(XBLOCK_SUB, RBLOCK)

            _tmp7 = tl.full([XBLOCK_SUB, RBLOCK], 0, tl.float32)
            for loop2 in range(loops2):
                r = loop2 * RBLOCK + base2[None, :]
                tmp0 = tl.load(in_ptr0 + (R_STRIDE * r + (X_STRIDE * x) + (Z_STRIDE * z)), None)
                tmp1 = tl.load(in_ptr1 + (R_STRIDE * r + (X_STRIDE * x) + (Z_STRIDE * z)), None)
                tmp2 = tmp0 + tmp1
                tmpSub = tmp2 - tmpMean
                _tmp7 = _tmp7 + tmpSub * tmpSub

            tmp7 = tl.sum(_tmp7, 1) / R
            tmpVar = tmp7.reshape(XBLOCK_SUB, 1).broadcast_to(XBLOCK_SUB, RBLOCK)
            tmpVar = tl.sqrt(tmpVar) + tl.full([XBLOCK_SUB, RBLOCK], 1e-5, tl.float32)
            for loop2 in range(loops2):
                r = loop2 * RBLOCK + base2[None, :]
                tmp0 = tl.load(in_ptr0 + (R_STRIDE * r + (X_STRIDE * x) + (Z_STRIDE * z)), None)
                tmp1 = tl.load(in_ptr1 + (R_STRIDE * r + (X_STRIDE * x) + (Z_STRIDE * z)), None)
                tmp2 = tmp0 + tmp1
                tmpSub = tmp2 - tmpMean
                tmpOut = tmpSub / tmpVar
                tl.store(out_ptr0 + (R_STRIDE * r + (X_STRIDE * x) + (Z_STRIDE * z)), tmpOut, None)

target_name = torch.npu.get_device_name(torch.npu.current_device())
guards = {"dummy":None}

import time
def test_addLayerNorm_triton_low(a, b):
    NBLOCKS = 32
    XBLOCK = a.shape[1] // NBLOCKS
    XBLOCK_SUB = min(64, XBLOCK)
    RBLOCK = min(32, a.shape[2])

    compile_opt = ASTSource(
        fn = test_addLayerNorm_loop,
        signature={0: '*fp32', 1: '*fp32', 2: '*fp32'},
        constants={3: XBLOCK, 4: XBLOCK_SUB, 5: RBLOCK,
                   6: a.shape[0], 7: a.shape[1], 8: a.shape[2],
                   9: a.stride()[0], 10: a.stride()[1], 11: a.stride()[2],
                   },
        attrs=AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=()),
    )

    hash_code = "tt_test_sum_" + str(time.perf_counter())

    compiled_func = guards.get(hash_code)
    if compiled_func == None:
        compiled_func = triton.compile(compile_opt, None, {"debug": True, "mix_mode": "aiv", "name": hash_code})
        guards[hash_code] = compiled_func

    value = torch.empty_strided((a.shape[0], a.shape[1], a.shape[2]),
                                (a.stride()[0], a.stride()[1], a.stride()[2]), dtype=torch.float32).npu()
    compiled_func[NBLOCKS, 1, 1](a, b, value)
    return value

def add_LayerNorm(a, b):
    x = a + b
    mean = torch.mean(x, dim = 2, keepdim = True)
    var = torch.mean((x - mean) ** 2, dim = 2, keepdim = True) + 1e-5
    y = (x - mean) / torch.sqrt(var)
    return y

if __name__ == "__main__":
    hidden_states = torch.randn((Z, X, Y), dtype = torch.float32).npu()
    add_layer = torch.randn((Z, X, Y), dtype = torch.float32).npu()

    comp_func = torch.compile(add_LayerNorm, backend="eager", dynamic=False)
    tritonOutput = test_addLayerNorm_triton_low(hidden_states, add_layer)
    output = comp_func(hidden_states, add_layer)
    print(tritonOutput[0:8, 0:2])
    print(output[0:8, 0:2])
    torch.testing.assert_close(tritonOutput, output, rtol=1e-4, atol=1e-4)
    print("Test code finish!")




