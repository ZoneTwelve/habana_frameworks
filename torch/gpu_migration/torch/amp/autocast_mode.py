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

from typing import Union

import torch

from habana_frameworks.torch.gpu_migration.core import _utils
from habana_frameworks.torch.gpu_migration.core._enums import Device
from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER
from habana_frameworks.torch.gpu_migration.torch import TorchModuleRegister


class autocast(torch.autocast, TorchModuleRegister):
    """
    .. py:gpumgrcall:: hpu_match

    Changes device_type to "hpu" and dtype to None, leaving the data type casting decision to hpu autocast engine.

    """

    @classmethod
    def _save_orig_func_gpu_migration(cls):
        return ["__init__"]

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __init__(
        self,
        device_type: str,
        dtype: Union[torch.dtype, None] = None,
        enabled: bool = True,
        cache_enabled: Union[bool, None] = None,
    ):
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None

        kwargs = {}
        if device_type.startswith(Device.cuda):
            kwargs["device_type"] = device_type.replace(Device.cuda, Device.hpu)
            # reset dtype to None, leave dtype selection to hpu autocast
            kwargs["dtype"] = None
        else:
            kwargs["device_type"] = device_type
            kwargs["dtype"] = dtype

        kwargs["enabled"] = enabled
        kwargs["cache_enabled"] = cache_enabled

        if device_type.startswith(Device.cuda):
            G_LOGGER.info(
                api_type="hpu_match",
                func_prefix="torch.autocast",
                old_args=log_args,
                new_call="torch.autocast.__init__({})".format(
                    _utils.kwargs_to_str(kwargs)
                ),
            )
        return autocast.call_parent_func("__init__", self, **kwargs)
