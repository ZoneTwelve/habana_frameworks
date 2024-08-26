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

from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER
from habana_frameworks.torch.gpu_migration.torch.nn import NNModuleRegister


class DataParallel(torch.nn.Module, NNModuleRegister):
    """
    .. py:gpumgrcall:: hpu_mismatch

    Raises NotImplementedError. Please use torch.nn.DistributedDataParallel instead.

    """

    # Descriptions:
    #     Currenly supported DataParallel devices only include cuda and xpu. Not supported on HPU. This API is added to avoid executing CUDA logic on HPU because device_type will be "cuda" if torch.cuda.is_available() is True.
    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        torch.nn.Module.__init__(self)

        G_LOGGER.info(
            api_type="hpu_mismatch",
            func_prefix="torch.nn",
            new_call="raise NotImplementedError",
        )

        raise NotImplementedError("DataParallel is currently not supported on HPU. Please use torch.nn.DistributedDataParallel instead.")
