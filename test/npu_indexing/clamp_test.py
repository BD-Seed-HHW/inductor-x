# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import torch
import torch_npu
import inductor_npu
import logging
import pytest
from test.testutils import OperatorType, TestUtils


class TestClamp(TestUtils):
    __TIME_LIMIT = 100
    __OPTYPE = OperatorType.POINTWISE
    torch._logging.set_logs(inductor=logging.DEBUG)

    # optimized function, auto timeout after __TIME_LIMIT seconds

    # @torch.compile(options={"aggressive_fusion": False})

    def op_calc(self, input, min=None, max=None):
        return input.clamp(min, max)


    # 在连续测试场景下,测试结果不稳定,建议单独重测批量测试未通过的 case
    # 若需测试更多数据类型，将dtype后面的list改成 ProtoTestCase._test_dtypes即可
    # 对indexing开关情况的测试需要用外部参数--npu_indexing=True/False完成

    @pytest.mark.timeout(__TIME_LIMIT)
    @pytest.mark.parametrize('shape', TestUtils._pointwise_demo_shapes)
    @pytest.mark.parametrize('dtype', ['float16', 'float32', 'bfloat16', 'int32', 'int64'])
    def test_pointwise_cases_minmax_is_tensor(self, shape, dtype, clear_cache):
        print(shape)
        print('npu_indexing= {}'.format(inductor_npu.config.enable_npu_indexing))
        min = self._generate_tensor(shape, dtype)
        max = self._generate_tensor(shape, dtype)

        first_element = self._generate_tensor(shape, dtype)

        std_result = self.op_calc(first_element, min=min, max=max)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, min=min, max=max)

        torch.testing.assert_close(std_result, inductor_result)

    @pytest.mark.timeout(__TIME_LIMIT)
    @pytest.mark.parametrize('shape', [(1,)])
    @pytest.mark.parametrize('dtype', ['float32'])
    def test_pointwise_cases_single_scalar(self, shape, dtype, clear_cache):
        print(shape)
        print('npu_indexing= {}'.format(inductor_npu.config.enable_npu_indexing))
        min = 0
        max = 100

        first_element = 200 * torch.rand(size=shape, dtype=eval('torch.' + dtype), device=torch.device("npu"))

        std_result = self.op_calc(first_element, min=min, max=max)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, min=min, max=max)
        torch.testing.assert_close(std_result, inductor_result)

    @pytest.mark.timeout(__TIME_LIMIT)
    @pytest.mark.parametrize('shape', [(1024, 32)])
    @pytest.mark.parametrize('dtype', ['int32'])
    @pytest.mark.skip(reason='not support yet')
    def test_pointwise_cases_minmax_is_number(self, shape, dtype, clear_cache):
        print(shape)
        print('npu_indexing= {}'.format(inductor_npu.config.enable_npu_indexing))
        min = 0
        max = 100

        first_element = self._generate_tensor(shape, dtype)

        std_result = self.op_calc(first_element, min=min, max=max)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, min=min, max=max)

        torch.testing.assert_close(std_result, inductor_result)

    @pytest.mark.timeout(__TIME_LIMIT)
    @pytest.mark.parametrize('shape', TestUtils._pointwise_demo_shapes)
    @pytest.mark.parametrize('dtype', ['float16', 'float32', 'bfloat16', 'int32', 'int64'])
    def test_pointwise_cases_max_only(self, shape, dtype, clear_cache):
        print(shape)
        print('npu_indexing= {}'.format(inductor_npu.config.enable_npu_indexing))
        max = 100

        first_element = self._generate_tensor(shape, dtype)

        std_result = self.op_calc(first_element, min=None, max=max)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, min=None, max=max)

        torch.testing.assert_close(std_result, inductor_result)

    @pytest.mark.timeout(__TIME_LIMIT)
    @pytest.mark.parametrize('shape', TestUtils._pointwise_demo_shapes)
    @pytest.mark.parametrize('dtype', ['float16', 'float32', 'bfloat16', 'int32', 'int64'])  
    def test_pointwise_cases_min_only(self, shape, dtype, clear_cache):
        print(shape)
        print('npu_indexing= {}'.format(inductor_npu.config.enable_npu_indexing))
        min = 0

        first_element = self._generate_tensor(shape, dtype)

        std_result = self.op_calc(first_element, min=min, max=None)

        compiled_op_calc = torch.compile(self.op_calc, backend="inductor")
        inductor_result = compiled_op_calc(first_element, min=min, max=None)

        torch.testing.assert_close(std_result, inductor_result)
if __name__ == '__main__':
    obj = TestClamp()
    obj.test_pointwise_cases_single_scalar((1,), 'float32', None)