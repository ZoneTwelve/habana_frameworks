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
from habana_frameworks.torch.gpu_migration.torch import TORCH_VERSION
from typing import Optional, Union

from . import ParallelModuleRegister
from torch.nn.parallel.distributed import _MixedPrecision


class DistributedDataParallel(
    torch.nn.parallel.DistributedDataParallel, ParallelModuleRegister
):
    """
    .. py:gpumgrcall:: hpu_match

    Sets device_ids and output_device to None.

    """

    @classmethod
    def _save_orig_func_gpu_migration(cls):
        return ["__init__"]

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)
    # TODO: Please remove else path once PT2.4 will be the default version
    if TORCH_VERSION.startswith('2.4'):
        def __init__(
            self,
            module,
            device_ids=None,
            output_device=None,
            dim=0,
            broadcast_buffers=True,
            process_group=None,
            bucket_cap_mb=None,
            find_unused_parameters=False,
            check_reduction=False,
            gradient_as_bucket_view=False,
            static_graph=False,
            delay_all_reduce_named_params=None,
            param_to_hook_all_reduce=None,
            mixed_precision: Optional[_MixedPrecision] = None,
            device_mesh=None,
            ):
            if not device_ids or not output_device:
                log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
                if log_args is not None:
                    log_args["module"] = "module"
                G_LOGGER.info(
                    api_type="hpu_match",
                    func_prefix="torch.nn.parallel.DistributedDataParallel",
                    old_args=log_args,
                    new_call="change device_ids and output_device to None",
                )

            device_ids = None
            output_device = None

            DistributedDataParallel.call_parent_func(
                "__init__",
                self,
                module,
                device_ids,
                output_device,
                dim,
                broadcast_buffers,
                process_group,
                bucket_cap_mb,
                find_unused_parameters,
                check_reduction,
                gradient_as_bucket_view,
                static_graph,
                delay_all_reduce_named_params,
                param_to_hook_all_reduce,
                mixed_precision,
                device_mesh,
            )
    else:
        def __init__(
            self,
            module,
            device_ids=None,
            output_device=None,
            dim=0,
            broadcast_buffers=True,
            process_group=None,
            bucket_cap_mb=25,
            find_unused_parameters=False,
            check_reduction=False,
            gradient_as_bucket_view=False,
            static_graph=False,
            delay_all_reduce_named_params=None,
            param_to_hook_all_reduce=None,
            mixed_precision: Optional[torch.nn.parallel.distributed._MixedPrecision] = None,
            device_mesh=None,
        ):
            if not device_ids or not output_device:
                log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
                if log_args is not None:
                    log_args["module"] = "module"
                G_LOGGER.info(
                    api_type="hpu_match",
                    func_prefix="torch.nn.parallel.DistributedDataParallel",
                    old_args=log_args,
                    new_call="change device_ids and output_device to None",
                )

            device_ids = None
            output_device = None

            DistributedDataParallel.call_parent_func(
                "__init__",
                self,
                module,
                device_ids,
                output_device,
                dim,
                broadcast_buffers,
                process_group,
                bucket_cap_mb,
                find_unused_parameters,
                check_reduction,
                gradient_as_bucket_view,
                static_graph,
                delay_all_reduce_named_params,
                param_to_hook_all_reduce,
                mixed_precision,
                device_mesh,
            )
