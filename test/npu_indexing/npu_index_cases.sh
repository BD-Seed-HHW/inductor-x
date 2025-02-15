#!/bin/bash

# ***********************************************************************
# Copyright: (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
# script for build and package waas framework
# ***********************************************************************

for entry in `ls ./*_test.py`
do
  export TORCH_COMPILE_DEBUG=0
  echo ""
  echo "start case ${entry} ++++++++++++++++++"
  echo ""
  python "$entry"
  if [ $? = 0 ]; then
      echo ""
      echo "done case ${entry} --------------------"
  else
      echo ""
      echo "failed case ${entry} --------------------"
      exit 1
  fi
  echo ""
done

echo "UT finished"
exit 0
