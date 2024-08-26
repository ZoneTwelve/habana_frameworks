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

from habana_frameworks.torch.gpu_migration.core.register import \
    BaseModuleRegister


class CudaAmpModuleRegister(BaseModuleRegister):
    @classmethod
    def get_module(cls):
        return torch.cuda.amp


from .grad_scaler import *

__all__ = ["GradScaler"]
