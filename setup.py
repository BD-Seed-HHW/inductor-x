# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from setuptools import setup, find_packages

setup(
    name='inductor_npu',
    version='0.1',
    packages=find_packages(exclude=("src", "test", "test2")),
    description='inductor_npu',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
