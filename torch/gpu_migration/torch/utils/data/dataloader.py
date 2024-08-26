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

from . import DataModuleRegister

from typing import (Any, Callable, Iterable, List, Optional, Sequence, TypeVar,
                    Union)

from habana_frameworks.torch.gpu_migration.core import _utils
from habana_frameworks.torch.gpu_migration.core._enums import Device
from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER
from habana_frameworks.torch.gpu_migration.torch import TORCH_VERSION
import torch


NoneType = type(None)


class DataLoader(torch.utils.data.DataLoader, DataModuleRegister):
    """
    .. py:gpumgrcall:: hpu_match

    If pin_memory is True and pin_memory_device is None/CUDA, sets pin_memory_device to "hpu".

    """

    # Description:
    #     In PyTorch, if pin_memory is set but pin_memory_device is not specified, default behaviour is CUDA device. For other backends, pin_memory_device must be specified.
    @classmethod
    def _save_orig_func_gpu_migration(cls) -> List:
        return ["__init__"]

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)



    def __init__(self, dataset: torch.utils.data.dataset.Dataset[TypeVar('T_co', covariant=True)], batch_size: Optional[int] = 1,
                shuffle: Optional[bool] = None, sampler: Union[torch.utils.data.sampler.Sampler, Iterable, None] = None,
                batch_sampler: Union[torch.utils.data.sampler.Sampler[List], Iterable[List], None] = None,
                num_workers: int = 0, collate_fn: Optional[Callable[[List[TypeVar('T')]], Any]] = None,
                pin_memory: bool = False, drop_last: bool = False,
                timeout: float = 0, worker_init_fn: Optional[Callable[[int], None]] = None,
                multiprocessing_context=None, generator=None,
                *, prefetch_factor: Union[int, NoneType]=None,
                persistent_workers: bool = False,
                pin_memory_device: str = ""):
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None

        new_pin_memory_device = pin_memory_device
        if pin_memory and pin_memory_device == "":
            new_pin_memory_device = Device.hpu
        elif pin_memory and pin_memory_device and str(pin_memory_device).startswith(Device.cuda):
            new_pin_memory_device = str(pin_memory_device).replace(
                Device.d_cuda, Device.d_hpu)

        if pin_memory_device != new_pin_memory_device and G_LOGGER.module_severity <= G_LOGGER.INFO:
            log_args['dataset'] = 'dataset'
            G_LOGGER.info(api_type="hpu_match", func_prefix="torch.utils.data.DataLoader", old_args=log_args, new_call="change pin_memory_device to {}".format(new_pin_memory_device))

        DataLoader.call_parent_func("__init__", self, dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout,
                                    worker_init_fn, multiprocessing_context, generator, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, pin_memory_device=new_pin_memory_device)
