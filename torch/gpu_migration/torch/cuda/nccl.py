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


import torch

from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER
from habana_frameworks.torch.gpu_migration.core.register import \
    BaseModuleRegister


class NCCLModuleRegister(BaseModuleRegister):
    @classmethod
    def get_module(cls):
        return torch.cuda.nccl


@NCCLModuleRegister.register_f()
def version():
    """
    .. py:gpumgrcall:: version.hpu_match

    Returns dummy NCCL version.

    """
    
    G_LOGGER.info(api_type="hpu_match", func_prefix="torch.cuda.nccl", new_call="Dummy")

    major = 2
    minor = 16
    patch = 5

    return (major, minor, patch)
