###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import torch

from habana_frameworks.torch.gpu_migration.core.register import \
    BaseModuleRegister
from habana_frameworks.torch.gpu_migration.torch import TORCH_VERSION


class DistributedModuleRegister(BaseModuleRegister):
    @classmethod
    def get_module(cls):
        return torch.distributed


from .distributed_c10d import *


__all__ = ["init_process_group", "barrier", "is_nccl_available", "get_backend"]

class DistributedLoggerModuleRegister(BaseModuleRegister):
    @classmethod
    def get_module(cls):
        return torch.distributed.c10d_logger

from .c10d_logger import _get_msg_dict

__all__.append("_get_msg_dict")

import habana_frameworks.torch.gpu_migration.torch.distributed.fsdp