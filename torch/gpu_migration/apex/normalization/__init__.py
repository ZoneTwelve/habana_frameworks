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
    - apex.normalization
"""
from types import ModuleType

import apex

from habana_frameworks.torch.gpu_migration.core.register import \
    BaseModuleRegister


class APEXNormModuleRegister(BaseModuleRegister):
    @classmethod
    def get_module(cls) -> ModuleType:
        return apex.normalization


class APEXNormFuncModuleRegister(BaseModuleRegister):
    @classmethod
    def get_module(cls) -> ModuleType:
        return apex.normalization.fused_layer_norm


from . import fused_layer_norm