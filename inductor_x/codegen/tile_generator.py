import copy
import pdb

import math

from torch._inductor.triton_heuristics import Config
from torch._inductor.utils import next_power_of_2
from .triton_utils import get_aligned_numel, byte_per_numel
# generate tiling configs
class TileGenerator:
    
    @staticmethod
    def aligned_numel(numel):
        aligned = next_power_of_2(numel)
        return aligned

    @staticmethod
    def get_byte_per_numel(dtype):
        if dtype is None :
            return 1
        return byte_per_numel[dtype]

    @staticmethod
    def valid_config(config, rnumel = 1):
        dtype = config["split_axis_dtype"]
        bytes = TileGenerator.get_byte_per_numel(dtype)
        max_numel = 16384 * 4 // bytes

        rblock = config["RBLOCK"] if "RBLOCK" in config else rnumel
        xblock_sub = config["XBLOCK_SUB"]
        if rblock * xblock_sub <= max_numel:
            return True

        return False

    # when rblock is low dim, need to maximize rblock
    @staticmethod
    def descend_xblock(rnumel, xblock, configs, cfg):
        dtype = cfg["split_axis_dtype"]
        bytes = TileGenerator.get_byte_per_numel(dtype)
        start_numel = 2048 // bytes
        # include rblock is too big, need to decend rblock first
        rblock = rnumel if rnumel > 0 else 1
        while (rblock > start_numel):
            newcfg = copy.deepcopy(cfg)
            newcfg["RBLOCK"] = rblock
            if TileGenerator.valid_config(newcfg):
                configs.append(Config(newcfg, num_warps=1, num_stages=1))
            rblock = rblock // 2
        cfg["RBLOCK"] = rblock
        xblock_sub = TileGenerator.aligned_numel(xblock)

        while True:
           newcfg = copy.deepcopy(cfg)
           newcfg["XBLOCK_SUB"] = xblock_sub
           if TileGenerator.valid_config(newcfg, rnumel=rblock):
               configs.append(Config(newcfg, num_warps=1, num_stages=1))
           xblock_sub = xblock_sub // 2
           if xblock_sub * rblock <= start_numel:
               break

    @staticmethod
    def descend_rblock(rnumel, xblock, configs, cfg):
        dtype = cfg["split_axis_dtype"]
        bytes = TileGenerator.get_byte_per_numel(dtype)
        start_numel = 4096 // bytes

        xblock_sub = start_numel if xblock > start_numel else xblock
        cfg["XBLOCK_SUB"] = xblock_sub
        rblock = rnumel
        while True:
            newcfg = copy.deepcopy(cfg)
            newcfg["RBLOCK"] = rblock
            if TileGenerator.valid_config(newcfg):
                 configs.append(Config(newcfg, num_warps=1, num_stages=1))
            rblock = rblock // 2
            if xblock_sub * rblock <= start_numel:
                break

    @staticmethod
    def descend_xblock_rblock(rnumel, xblock, configs, cfg) :
        dtype = cfg["split_axis_dtype"]
        bytes = TileGenerator.get_byte_per_numel(dtype)
        start_numel = 4096 // bytes
        # Depending on the number of bytes available to the hardware UB,
        # 4096 bytes is an appropriate empirical value for an intra-core split.
        # Rule: xblock_sub * rblock <= start_numel
        end_numel = math.floor(math.sqrt(start_numel))

        xblock = next_power_of_2(xblock)
        rnumel = next_power_of_2(rnumel)

        xblock_sub = xblock  if xblock > start_numel else xblock
        rblock = start_numel if rnumel > start_numel else rnumel

        rblock_is_biggerr = rblock > xblock_sub

        if xblock_sub * rblock <= start_numel :
            newcfg = copy.deepcopy(cfg)
            newcfg["XBLOCK_SUB"] = xblock_sub
            newcfg["RBLOCK"] = rblock
            if TileGenerator.valid_config(newcfg):
                configs.append(Config(newcfg, num_warps=1, num_stages=1))

        if rblock_is_biggerr:
            while rblock > xblock_sub and xblock_sub * rblock > start_numel:
                newcfg = copy.deepcopy(cfg)
                newcfg["RBLOCK"] = rblock
                xblock_sub = xblock
                if TileGenerator.valid_config(newcfg):
                    configs.append(Config(newcfg, num_warps=1, num_stages=1))
                rblock = rblock // 2
        else :
            while rblock < xblock_sub and xblock_sub * rblock > start_numel:
                newcfg = copy.deepcopy(cfg)
                newcfg["XBLOCK_SUB"] = xblock_sub
                if TileGenerator.valid_config(newcfg):
                    configs.append(Config(newcfg, num_warps=1, num_stages=1))
                xblock_sub = xblock_sub // 2

        while xblock_sub * rblock > start_numel :
            newcfg = copy.deepcopy(cfg)
            newcfg["XBLOCK_SUB"] = xblock_sub
            newcfg["RBLOCK"] = rblock
            if TileGenerator.valid_config(newcfg):
                configs.append(Config(newcfg, num_warps=1, num_stages=1))
            if xblock_sub >= end_numel:
                xblock_sub = xblock_sub // 2
            if rblock >= end_numel:
                rblock = rblock // 2

    @staticmethod
    def nearest_power_of_2(n):
        big = next_power_of_2(n)
        small = big // 2
        return big if (big - n) < (n - small) else small
