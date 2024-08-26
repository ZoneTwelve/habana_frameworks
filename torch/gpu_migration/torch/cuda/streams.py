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

from . import CudaModuleRegister


class Stream(torch.hpu.Stream, CudaModuleRegister):
    """
    .. py:gpumgrcall:: hpu_match

    Maps torch.cuda.Stream to torch.hpu.Stream.
    """

    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __new__(cls, device=None, priority=0, **kwargs):
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None

        s = super().__new__(cls, device=device, priority=priority, **kwargs)
        s.cuda_stream = s.hpu_stream

        G_LOGGER.info(
            api_type="hpu_match",
            func_prefix="torch.cuda",
            old_args=log_args,
            new_call="return torch.hpu.Stream({}, {}, {})".format(
                device, priority, kwargs
            ),
        )
        return s


class ExternalStream(torch.hpu.Stream, CudaModuleRegister):
    """
    .. py:gpumgrcall:: hpu_modified

    Maps torch.cuda.Stream.ExternalStream to torch.hpu.Stream.

    """

    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)


class Event(torch.hpu.Event, CudaModuleRegister):
    """
    .. py:gpumgrcall:: functional_mismatch

    Maps torch.cuda.Event to torch.hpu.Event.

    Descriptions:
        blocking ignored, cause Synapse event do not have busy waiting
    """

    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __init__(self, enable_timing=False, blocking=False, interprocess=False):
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
        G_LOGGER.info(
            api_type="functional_mismatch",
            func_prefix="torch.cuda",
            old_args=log_args,
            new_call="torch.hpu.Event({})".format(enable_timing),
        )
        super().__init__(enable_timing)
