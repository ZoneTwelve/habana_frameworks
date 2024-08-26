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

from ._VariableFunctions import *

import torch
import sys
import warnings

from torch.types import _int
from habana_frameworks.torch.gpu_migration.core import _utils
from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER

def _cuda_getCompiledVersion():
    return torch.version.cuda

def _cuda_setStream(stream_id: _int, device_index: _int, device_type: _int) -> None:
    """
    .. py:gpumgrcall:: _cuda_setStream.hpu_mismatch

    Inactive Call.

    """
    # TODO: SW-176566
    G_LOGGER.info(
        api_type="hpu_mismatch",
        func_prefix="torch._C",
        new_call="Dummy",
    )
    warnings.warn("_cuda_setStream (set stream by stream id) is not supported on HPU")

def _cuda_setDevice(device: _int) -> None:
    """
    .. py:gpumgrcall:: _cuda_setDevice.hpu_match

    Maps torch._C._cuda_setDevice to torch.hpu.set_device.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    device = _utils.convert_device_arg(device)
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch._C",
        old_args=log_args,
        new_call="torch.hpu.set_device({})".format(device),
    )
    torch.hpu.set_device(device)

sys.modules["torch._C"]._cuda_getCompiledVersion = _cuda_getCompiledVersion
sys.modules["torch._C"]._cuda_setStream = _cuda_setStream
sys.modules["torch._C"]._cuda_setDevice = _cuda_setDevice