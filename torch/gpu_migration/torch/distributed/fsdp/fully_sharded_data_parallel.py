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


from typing import Callable, Iterable, Optional, Union

import torch
import torch.nn as nn
import torch.distributed
import torch.distributed.fsdp
from torch.distributed.fsdp._init_utils import (
    HYBRID_SHARDING_STRATEGIES,
    ProcessGroupType,
)
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed._tensor import DeviceMesh
from torch.distributed.fsdp.wrap import CustomPolicy, ModuleWrapPolicy
from torch.distributed.fsdp import FullyShardedDataParallel

from habana_frameworks.torch.gpu_migration.core._enums import Device
from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER

from . import FSDPModuleRegister


class FullyShardedDataParallel(FullyShardedDataParallel, FSDPModuleRegister):
    """
    .. py:gpumgrcall:: FullyShardedDataParallel.hpu_match

    Changes device_id arguments from “cuda” to “hpu”.

    """

    @classmethod
    def _save_orig_func_gpu_migration(cls):
        return ["__init__"]

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __init__(
        self,
        module: nn.Module,
        process_group: ProcessGroupType = None,
        sharding_strategy: Optional[ShardingStrategy] = None,
        cpu_offload: Optional[CPUOffload] = None,
        auto_wrap_policy: Optional[
            Union[Callable, ModuleWrapPolicy, CustomPolicy]
        ] = None,
        backward_prefetch: Optional[BackwardPrefetch] = BackwardPrefetch.BACKWARD_PRE,
        mixed_precision: Optional[MixedPrecision] = None,
        ignored_modules: Optional[Iterable[torch.nn.Module]] = None,
        param_init_fn: Optional[Callable[[nn.Module], None]] = None,
        device_id: Optional[Union[int, torch.device]] = None,
        sync_module_states: bool = False,
        forward_prefetch: bool = False,
        limit_all_gathers: bool = True,
        use_orig_params: bool = False,
        ignored_states: Union[
            Optional[Iterable[torch.nn.Parameter]], Optional[Iterable[torch.nn.Module]]
        ] = None,
        device_mesh: Optional[DeviceMesh] = None,
    ):
        d_id = device_id
        if device_id is None:
            d_id = torch.device(Device.hpu, torch.hpu.current_device())
        else:
            if isinstance(device_id, str):
                if device_id.startswith(Device.cuda):
                    d_id = torch.device(Device.hpu, torch.hpu.current_device())
                elif device_id.isdigit():
                    d_id = torch.device(Device.hpu, torch.hpu.current_device())
            elif isinstance(device_id, torch.device):
                if str(device_id).startswith(Device.cuda):
                    d_id = torch.device(Device.hpu, torch.hpu.current_device())
            elif isinstance(device_id, int):
                d_id = torch.device(Device.hpu, torch.hpu.current_device())

        if d_id != device_id:
            device_id = d_id
            G_LOGGER.info(
                api_type="hpu_match",
                func_prefix="torch.distributed.fsdp.FullyShardedDataParallel",
                new_call="change device_id to torch.device(\"hpu\", torch.hpu.current_device())",
            )

        return FullyShardedDataParallel.call_parent_func(
            "__init__",
            self,
            module,
            process_group,
            sharding_strategy,
            cpu_offload,
            auto_wrap_policy,
            backward_prefetch,
            mixed_precision,
            ignored_modules,
            param_init_fn,
            device_id=device_id,
            sync_module_states=sync_module_states,
            forward_prefetch=forward_prefetch,
            limit_all_gathers=limit_all_gathers,
            use_orig_params=use_orig_params,
            ignored_states=ignored_states,
            device_mesh=device_mesh,
        )
