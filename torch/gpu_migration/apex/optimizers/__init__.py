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

"""
The module to be replaced:
    - apex.optimizers
"""
from types import ModuleType

import apex
import warnings

from habana_frameworks.torch.gpu_migration.core.register import \
    BaseModuleRegister


class APEXOptimizersModuleRegister(BaseModuleRegister):
    @classmethod
    def get_module(cls) -> ModuleType:
        return apex.optimizers

from habana_frameworks.torch.utils.internal import is_lazy

from .fused_adagrad import FusedAdagrad
from .fused_adam import FusedAdam
from .fused_lamb import FusedLAMB
from .fused_sgd import FusedSGD


__all__ = ["FusedSGD", "FusedAdam", "FusedLAMB", "FusedAdagrad"]
