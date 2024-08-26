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

from datetime import timedelta
from typing import Any, Optional, Union

import torch
from torch._C._distributed_c10d import _DEFAULT_PG_TIMEOUT, ProcessGroup, Store
from torch.distributed import Backend, GroupMember

from habana_frameworks.torch.gpu_migration.core._enums import DistBackend, Device
from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER
from habana_frameworks.torch.gpu_migration.torch import TORCH_VERSION

from . import DistributedModuleRegister


@DistributedModuleRegister.register_f("init_process_group")
def init_process_group(
    backend: Optional[str] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = "",
    pg_options: Optional[Any] = None,
    device_id: Optional[torch.device] = None,
) -> None:
    """
    .. py:gpumgrcall:: init_process_group.hpu_match

    Changes backend from nccl to hccl.

    """
    if backend is None or backend == DistBackend.cuda:
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
        backend = DistBackend.hpu
        G_LOGGER.info(api_type="hpu_match", func_prefix="torch.distributed",
                    old_args=log_args, new_call="change backend to {}".format(backend))

    return DistributedModuleRegister.call_parent_func("init_process_group", backend, init_method, timeout, world_size, rank, store, group_name, pg_options)


@DistributedModuleRegister.register_f("get_backend")
def get_backend(group: Optional[ProcessGroup] = None) -> Backend:
    """
    .. py:gpumgrcall:: get_backend.hpu_match

    Returns nccl as a backend.

    """
    # Description:
    #     >>> # User may check return value, for example in YOLOX:
    #     >>> backend = dist.get_backend(group)
    #     >>> assert backend in ["gloo", "nccl"]
    backend = DistributedModuleRegister.call_parent_func("get_backend", group)

    if backend == "hccl":
        backend = "nccl"
        G_LOGGER.info(
            api_type="hpu_match",
            func_prefix="torch.distributed",
            new_call="change return value from hccl to nccl",
        )

    return backend


@DistributedModuleRegister.register_f("barrier")
def barrier(group=GroupMember.WORLD, async_op=False, device_ids=None):
    """
    .. py:gpumgrcall:: barrier.hpu_match

    Ignores device_ids, which is valid only for NCCL backend.
    """
    #Changed Args:
    #    - device_ids: set to None if device_ids is a list of device/GPU ids.

    #Descriptions:
    #    device_ids is valid only for NCCL backend.
    if device_ids:
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None

        device_ids = None

        G_LOGGER.info(
            api_type="hpu_match",
            func_prefix="torch.distributed",
            old_args=log_args,
            new_call=(
                "torch.distributed.barrier(group={}, async_op={}, device_ids=None)"
                .format(group, async_op)
            ),
        )

    return DistributedModuleRegister.call_parent_func(
        "barrier", group, async_op, device_ids
    )


@DistributedModuleRegister.register_f()
def is_nccl_available() -> bool:
    """
    .. py:gpumgrcall:: is_nccl_available.hpu_match

    Checks import habana_frameworks.torch.distributed.hccl. If success, return True; else return False.

    """
    try:
        import habana_frameworks.torch.distributed.hccl

        G_LOGGER.info(
            api_type="hpu_match",
            func_prefix="torch.distributed",
            new_call="change return value to True",
        )

        return True
    except ImportError:
        return False

# TODO: Please remove else path once PT2.4 will be the default version
if TORCH_VERSION.startswith('2.4'):
    @DistributedModuleRegister.register_f("new_group")
    def new_group(
        ranks=None,
        timeout=None,
        backend=None,
        pg_options=None,
        use_local_synchronization=False,
        group_desc=None
    ):
        """
        .. py:gpumgrcall:: new_group.hpu_match

        Changes backend from nccl to hccl.

        """

        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None

        if backend is not None:
            if DistBackend.cuda in backend or Device.cuda in backend:
                backend = backend.replace(DistBackend.cuda, DistBackend.hpu)
                backend = backend.replace(Device.cuda, Device.hpu)
                
                G_LOGGER.info(api_type="hpu_match", func_prefix="torch.distributed",
                            old_args=log_args, new_call="change backend to {}".format(backend))

        return DistributedModuleRegister.call_parent_func("new_group",
                                                        ranks=ranks,
                                                        timeout=timeout,
                                                        backend=backend,
                                                        pg_options=pg_options,
                                                        )
else:
    @DistributedModuleRegister.register_f("new_group")
    def new_group(
        ranks=None,
        timeout=None,
        backend=None,
        pg_options=None,
        use_local_synchronization=False,
    ):
        """
        .. py:gpumgrcall:: new_group.hpu_match

        Changes backend from nccl to hccl.

        """

        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None

        if backend is not None:
            if DistBackend.cuda in backend or Device.cuda in backend:
                backend = backend.replace(DistBackend.cuda, DistBackend.hpu)
                backend = backend.replace(Device.cuda, Device.hpu)
                
                G_LOGGER.info(api_type="hpu_match", func_prefix="torch.distributed",
                            old_args=log_args, new_call="change backend to {}".format(backend))

        return DistributedModuleRegister.call_parent_func("new_group",
                                                        ranks=ranks,
                                                        timeout=timeout,
                                                        backend=backend,
                                                        pg_options=pg_options,
                                                        )