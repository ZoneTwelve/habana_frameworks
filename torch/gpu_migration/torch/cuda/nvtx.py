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

from habana_frameworks.torch.gpu_migration.core.register import BaseModuleRegister
from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER
from contextlib import contextmanager

from . import CudaModuleRegister


class CudaNvtxModuleRegister(BaseModuleRegister):
    @classmethod
    def get_module(cls):
        import torch.cuda.nvtx

        return torch.cuda.nvtx


@CudaNvtxModuleRegister.register_f()
@contextmanager
def range(msg, *args, **kwargs):
    """
    .. py:gpumgrcall:: range.hpu_mismatch

    Inactive Call.

    """
    # Context manager / decorator that pushes an NVTX range at the beginning of its scope, and pops it at the end. Not applicable on HPU.
    G_LOGGER.info(
        api_type="hpu_mismatch", func_prefix="torch.cuda.nvtx", new_call="Dummy"
    )
    yield
