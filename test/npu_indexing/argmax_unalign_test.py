# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import torch
import torch_npu
import sys
sys.path.append("../..")
import inductor_npu

import logging
import pytest
# from .testutils import OperatorType, TestUtils
inductor_npu.config.enable_npu_indexing = True
class TestMaxWithIndex():
    __TIME_LIMIT = 100
    # __OPTYPE = OperatorType.REDUCTION
    #torch._logging.set_logs(inductor=logging.DEBUG)

    # optimized function, auto timeout after __TIME_LIMIT seconds

    # @torch.compile(options={"aggressive_fusion": False})

    def op_calc(self, input_element, dim):
        return torch.argmax(input_element, dim)

    _reduction_extest_shape4d_all = [(8, 8, 1024, 2048), (8, 8, 2048, 1024), (8, 1024, 2048, 8), (2048, 8, 1024, 8),
                                     (2048, 1024, 8, 8)]
    _reduction_extest_dim4d_one = [-1]

    _reduction_extest_shape4d_one = [(8, 16, 1024, 64)]
    _reduction_extest_dim4d_all = [0, 1, 2, 3]
    # 在连续测试场景下,测试结果不稳定,建议单独重测批量测试未通过的 case
    # 若需测试更多数据类型，将dtype后面的list改成 ProtoTestCase._test_dtypes即可
    # 对indexing开关情况的测试需要用外部参数--npu_indexing=True/False完成

    #should be implemented when __OPTYPE is OperatorType.REDUCTION
    @pytest.mark.timeout(__TIME_LIMIT)
    @pytest.mark.parametrize('shape', [(513, 64)])
    @pytest.mark.parametrize('dim', [-1 ])
    @pytest.mark.parametrize('dtype', ['float32'])
    def test_reduction_cases(self, shape, dim, dtype):
        print(f"shape= {shape}")
        print(f"dim= {dim}")
        print(f"dtype= {dtype}")
        print('npu_indexing= {}'.format(inductor_npu.config.enable_npu_indexing))
        input_element = (torch.randn(size=shape, dtype=eval('torch.' + dtype)) * 2000).npu()

        std_argmax = self.op_calc(input_element, dim)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_argmax = compiled_op_calc(input_element, dim)
        #print(std_argmax)
        print("eager:",std_argmax.dtype)
        #print(inductor_argmax)
        print("inductor", inductor_argmax.dtype)
        torch.testing.assert_close(std_argmax, inductor_argmax, rtol=1e-2, atol=1e-2)
if __name__ == "__main__":
    #size = (512, 64)
    size = (513, 1024)
    #size = (512, 65) #no_test
    # size1 = ( 513, 65) #no_test
    # size2 = ( 512, 4) #
    test = TestMaxWithIndex()
    test.test_reduction_cases(size, -1, 'float32')
    print("data validation passed")