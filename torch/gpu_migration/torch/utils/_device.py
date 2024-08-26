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

from typing import List
from . import DeviceContextModuleRegister
from torch.utils._device import _device_constructors
from habana_frameworks.torch.gpu_migration.core import _utils
from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER
import torch

class DeviceContext(torch.utils._device.DeviceContext, DeviceContextModuleRegister):
    """
    .. py:gpumgrcall:: hpu_match

    Changes device_type of "cuda" to "hpu" in Device Context ('with' statement).

    """
    @classmethod
    def _save_orig_func_gpu_migration(cls) -> List:
        return ["__init__"]

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __init__(self, device):
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
        device_type = _utils.convert_device_arg(device)
        if type(device_type) is str:
            device_index = _utils.get_device_index(device)
            self.device = torch.device(device_type, device_index)
        else:
             self.device = torch.device(device_type)
        G_LOGGER.info(api_type="hpu_match",
                func_prefix="torch.utils.DeviceContext",
                old_args=log_args,
                new_call=f"torch.utils.DeviceContext.__init__(device={self.device})"
            )

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        parent_func_list = list(DeviceContextModuleRegister.parentFunc.values())
        if (func in _device_constructors() or func in parent_func_list) and kwargs.get('device') is None:
            kwargs['device'] = self.device
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
        G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.utils.DeviceContext.__torch_function__(func={}, types={}, args={}, kwargs={{{}}})".format(func, types, args, _utils.kwargs_to_str(kwargs)),
    )
        return func(*args, **kwargs)
