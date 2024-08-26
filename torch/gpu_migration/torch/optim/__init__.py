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

from habana_frameworks.torch.gpu_migration.core.register import \
    BaseModuleRegister

import torch

class TorchOptimizersModuleRegister(BaseModuleRegister):
    @classmethod
    def get_module(cls) -> ModuleType:
        return torch.optim


from .adadelta import Adadelta
from .adam import Adam
from .rmsprop import RMSprop
from .sgd import SGD