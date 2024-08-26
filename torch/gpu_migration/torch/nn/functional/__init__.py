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

from habana_frameworks.torch.utils.internal import is_lazy

class FunctionalModuleRegister(BaseModuleRegister):
    @classmethod
    def get_module(cls):
        return torch.nn.functional


if is_lazy():
    from .functional import scaled_dot_product_attention
    __all__ = ["scaled_dot_product_attention"]

