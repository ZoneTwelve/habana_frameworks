###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################


from types import ModuleType

from habana_frameworks.torch.gpu_migration.core.register import BaseModuleRegister
from habana_frameworks.torch.gpu_migration.torch import TORCH_VERSION

import torch


class FSDPModuleRegister(BaseModuleRegister):
    @classmethod
    def get_module(cls) -> ModuleType:
        return torch.distributed.fsdp


from .fully_sharded_data_parallel import FullyShardedDataParallel

__all__ = ["FullyShardedDataParallel"]
