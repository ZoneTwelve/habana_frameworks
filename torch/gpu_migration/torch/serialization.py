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
import os

import torch

from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER
from habana_frameworks.torch.gpu_migration.core._enums import Device
from habana_frameworks.torch.gpu_migration.torch import TORCH_VERSION
from typing import Any, BinaryIO, Callable, Dict, Optional, Union, IO
from typing_extensions import TypeAlias
from torch.types import Storage
FILE_LIKE: TypeAlias = Union[str, os.PathLike, BinaryIO, IO[bytes]]
if TORCH_VERSION.startswith('2.4'):
    MAP_LOCATION: TypeAlias = Optional[Union[Callable[[Storage, str], Storage], torch.device, str, Dict[str, str]]]
else:
    MAP_LOCATION: TypeAlias = Optional[Union[Callable[[torch.Tensor, str], torch.Tensor], torch.device, str, Dict[str, str]]]
    
from . import TorchModuleRegister

# TODO: Please remove else path once PT2.4 will be the default version
if TORCH_VERSION.startswith('2.4'):
    @TorchModuleRegister.register_f("load")
    def load(
         f: FILE_LIKE,
         map_location: MAP_LOCATION = None,
         pickle_module: Any = None,
         *,
         weights_only: Optional[bool] = None,
         mmap: Optional[bool] = None,
         **pickle_load_args: Any) -> Any:
        """
        .. py:gpumgrcall:: load.hpu_match

        Changes map_location to enable HPU mapping instead of CUDA mapping.

        """

        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None

        map_location_new = map_location

        if str(map_location).startswith(Device.cuda):
                    map_location_new = str(map_location).replace(Device.cuda, Device.hpu)
        elif ((isinstance(map_location, dict) and str(list(map_location.values())[0]).startswith(Device.cuda))):
            map_location_new = {}
            for device in map_location:
                map_location_new[str(device).replace(Device.cuda, Device.hpu)] = str(map_location[device]).replace(Device.cuda, Device.hpu)

        if map_location_new is not None:
            G_LOGGER.info(
                api_type="hpu_match",
                func_prefix="torch",
                old_args=log_args,
                new_call="torch.load(f={}, map_location={}, pickle_module={}, weights_only={}, mmap={}, pickle_load_args={})".format(
                    f, map_location_new, pickle_module, weights_only, mmap, str(pickle_load_args)),
            )

        return TorchModuleRegister.call_parent_func("load", f, map_location_new, pickle_module, weights_only=weights_only, mmap=mmap, **pickle_load_args)
else:
    @TorchModuleRegister.register_f("load")
    def load(
        f: FILE_LIKE,
        map_location: MAP_LOCATION = None,
        pickle_module: Any = None,
        *,
        weights_only: bool = False,
        mmap: Optional[bool] = None,
        **pickle_load_args: Any) -> Any:
        """
        .. py:gpumgrcall:: load.hpu_match

        Changes map_location to enable HPU mapping instead of CUDA mapping.

        """

        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None

        map_location_new = map_location

        if str(map_location).startswith(Device.cuda):
                    map_location_new = str(map_location).replace(Device.cuda, Device.hpu)
        elif ((isinstance(map_location, dict) and str(list(map_location.values())[0]).startswith(Device.cuda))):
            map_location_new = {}
            for device in map_location:
                map_location_new[str(device).replace(Device.cuda, Device.hpu)] = str(map_location[device]).replace(Device.cuda, Device.hpu)

        if map_location_new is not None:
            G_LOGGER.info(
                api_type="hpu_match",
                func_prefix="torch",
                old_args=log_args,
                new_call="torch.load(f={}, map_location={}, pickle_module={}, weights_only={}, mmap={}, pickle_load_args={})".format(
                    f, map_location_new, pickle_module, weights_only, mmap, str(pickle_load_args)),
            )

        return TorchModuleRegister.call_parent_func("load", f, map_location_new, pickle_module, weights_only=weights_only, mmap=mmap, **pickle_load_args)
