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
import warnings
from typing import Any, Dict, Tuple, Union

import torch
from torch.types import Device

from habana_frameworks.torch.gpu_migration.core import _utils
from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER

from . import CudaModuleRegister


@CudaModuleRegister.register_f()
def empty_cache() -> None:
    """
    .. py:gpumgrcall:: empty_cache.hpu_mismatch

    Inactive Call.

    """
    # Releases all unoccupied cached memory currently held by the caching allocator. Not suppoted on HPU.
    G_LOGGER.info(api_type="hpu_mismatch", func_prefix="torch.cuda", new_call="Dummy")
    warnings.warn(
        "No need to call empty_cache on HPU. It manages the memory internally in an"
        " efficient way."
    )


@CudaModuleRegister.register_f()
def list_gpu_processes(device: Union[Device, int] = None) -> str:
    """
    .. py:gpumgrcall:: list_gpu_processes.hpu_modified

    Prints out the running processes and their HPU memory use by calling torch.hpu.memory_reserved for the given device.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    device = _utils.convert_device_arg(device)
    G_LOGGER.info(
        api_type="hpu_modified",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="habana_frameworks.torch.gpu_migration.torch.hpu.list_gpu_processes(device={})".format(
            device
        ),
    )

    process_id = os.getpid()
    device_name = torch.hpu._get_device_index(device, optional=True)
    used_mem = torch.hpu.memory_reserved()

    return (
        f"HPU:{device_name}\nprocess {process_id} uses {used_mem/1024/1024:.3f} MB HPU"
        " memory"
    )


@CudaModuleRegister.register_f()
def mem_get_info(device: Union[Device, int] = None) -> Tuple[int, int]:
    """
    .. py:gpumgrcall:: mem_get_info.hpu_match

    Maps torch.cuda.mem_get_info to torch.hpu.mem_get_info.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    device = _utils.convert_device_arg(device)
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.hpu.mem_get_info(device={})".format(device),
    )

    return torch.hpu.mem_get_info(device)


@CudaModuleRegister.register_f()
def memory_stats(device: Union[Device, int] = None) -> Dict[str, Any]:
    """
    .. py:gpumgrcall:: memory_stats.hpu_match

    Maps torch.cuda.memory_stats to torch.hpu.memory_stats.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    device = _utils.convert_device_arg(device)
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.hpu.memory_stats(device={})".format(device),
    )

    return torch.hpu.memory_stats(device)


@CudaModuleRegister.register_f()
def memory_summary(device: Union[Device, int] = None, abbreviated: bool = False) -> str:
    """
    .. py:gpumgrcall:: memory_summary.hpu_match

    Maps torch.cuda.memory_summary to torch.hpu.memory_summary.
    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    device = _utils.convert_device_arg(device)
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.hpu.memory_summary(device={})".format(device),
    )

    return torch.hpu.memory_summary(device)


@CudaModuleRegister.register_f()
def memory_snapshot():
    """
    .. py:gpumgrcall:: memory_snapshot.hpu_mismatch

    Inactive Call.

    """
    G_LOGGER.info(api_type="hpu_mismatch", func_prefix="torch.cuda", new_call="Dummy")

    warnings.warn("memory_snapshot is not supported on HPU.")


@CudaModuleRegister.register_f()
def memory_allocated(device: Union[Device, int] = None) -> int:
    """
    .. py:gpumgrcall:: memory_allocated.hpu_match

    Maps torch.cuda.memory_allocated to torch.hpu.memory_allocated.
    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    device = _utils.convert_device_arg(device)
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.hpu.memory_allocated(device={})".format(device),
    )

    return torch.hpu.memory_allocated(device)


@CudaModuleRegister.register_f()
def max_memory_allocated(device: Union[Device, int] = None) -> int:
    """
    .. py:gpumgrcall:: max_memory_allocated.hpu_match

    Maps torch.cuda.max_memory_allocated to torch.hpu.max_memory_allocated.
    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    device = _utils.convert_device_arg(device)
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.hpu.max_memory_allocated(device={})".format(device),
    )

    return torch.hpu.max_memory_allocated(device)


@CudaModuleRegister.register_f()
def reset_max_memory_allocated(device: Union[Device, int] = None) -> None:
    """
    .. py:gpumgrcall:: reset_max_memory_allocated.hpu_match

    Maps torch.cuda.reset_max_memory_allocated to torch.hpu.reset_max_memory_allocated.
    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    device = _utils.convert_device_arg(device)
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.hpu.reset_peak_memory_stats(device={})".format(device),
    )

    return torch.hpu.reset_peak_memory_stats(device)


@CudaModuleRegister.register_f()
def memory_reserved(device: Union[Device, int] = None) -> int:
    """
    .. py:gpumgrcall:: memory_reserved.hpu_match

    Maps torch.cuda.memory_reserved to torch.hpu.memory_reserved.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    device = _utils.convert_device_arg(device)
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.hpu.memory_reserved(device={})".format(device),
    )

    return torch.hpu.memory_reserved(device)


@CudaModuleRegister.register_f()
def max_memory_reserved(device: Union[Device, int] = None) -> int:
    """
    .. py:gpumgrcall:: max_memory_reserved.hpu_match

    Maps torch.cuda.max_memory_reserved to torch.hpu.max_memory_reserved.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    device = _utils.convert_device_arg(device)
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.hpu.max_memory_reserved(device={})".format(device),
    )

    return torch.hpu.max_memory_reserved(device)


@CudaModuleRegister.register_f()
def set_per_process_memory_fraction(
    fraction, device: Union[Device, int] = None
) -> None:
    """
    .. py:gpumgrcall:: set_per_process_memory_fraction.hpu_mismatch

    Inactive Call.

    """
    # Set memory fraction for a process. Not supported on HPU.
    G_LOGGER.info(api_type="hpu_mismatch", func_prefix="torch.cuda", new_call="Dummy")

    warnings.warn("set_per_process_memory_fraction is not supported on HPU.")


@CudaModuleRegister.register_f()
def memory_cached(device: Union[Device, int] = None) -> int:
    """
    .. py:gpumgrcall:: memory_cached.hpu_match

    Maps torch.cuda.memory_cached to torch.hpu.memory_reserved.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    device = _utils.convert_device_arg(device)
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.hpu.memory_reserved(device={})".format(device),
    )

    return torch.hpu.memory_reserved(device)


@CudaModuleRegister.register_f()
def max_memory_cached(device: Union[Device, int] = None) -> int:
    """
    .. py:gpumgrcall:: max_memory_cached.hpu_match

    Maps torch.cuda.max_memory_cached to torch.hpu.max_memory_reserved.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    device = _utils.convert_device_arg(device)
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.hpu.max_memory_reserved(device={})".format(device),
    )

    return torch.hpu.max_memory_reserved(device)


@CudaModuleRegister.register_f()
def reset_max_memory_cached(device: Union[Device, int] = None) -> None:
    """
    .. py:gpumgrcall:: reset_max_memory_cached.hpu_mismatch

    Inactive Call.

    """
    # Resets the starting point in tracking maximum memory managed by the caching allocator for a given device. Not supported on HPU.
    G_LOGGER.info(api_type="hpu_mismatch", func_prefix="torch.cuda", new_call="Dummy")

    warnings.warn(
        "reset_max_memory_cached is not supported on HPU. The memory cache size is"
        " constant on HPU."
    )


@CudaModuleRegister.register_f()
def reset_peak_memory_stats(device: Union[Device, int] = None) -> None:
    """
    .. py:gpumgrcall:: reset_peak_memory_stats.hpu_match

    Maps torch.cuda.reset_peak_memory_stats to torch.hpu.reset_peak_memory_stats.

    Description:
        Resets the peak stats tracked by the memory allocator.
    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    device = _utils.convert_device_arg(device)
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.hpu.reset_peak_memory_stats(device={})".format(device),
    )

    return torch.hpu.reset_peak_memory_stats(device)


@CudaModuleRegister.register_f()
def caching_allocator_alloc(size, device: Union[Device, int] = None, stream=None):
    """
    .. py:gpumgrcall:: caching_allocator_alloc.hpu_mismatch

    Inactive Call.

    """
    # Description:
    #    Performs a memory allocation using the memory allocator. Not supported on HPU.
    G_LOGGER.info(api_type="hpu_mismatch", func_prefix="torch.cuda", new_call="Dummy")

    warnings.warn("caching_allocator_alloc is not supported on HPU.")


@CudaModuleRegister.register_f()
def caching_allocator_delete(mem_ptr):
    """
    .. py:gpumgrcall:: caching_allocator_delete.hpu_mismatch

    Inactive Call.

    """
    # Deletes memory allocated using the memory allocator. Not supported on HPU.
    G_LOGGER.info(api_type="hpu_mismatch", func_prefix="torch.cuda", new_call="Dummy")

    warnings.warn("caching_allocator_delete is not supported on HPU.")
