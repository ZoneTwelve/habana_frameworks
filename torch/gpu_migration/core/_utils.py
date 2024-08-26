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

from typing import Any, Optional

import torch

from habana_frameworks.torch.gpu_migration.core._enums import Device


def convert_device_arg(device):
    if isinstance(device, str):
        if device.startswith(Device.cuda):
            device = device.replace(Device.cuda, Device.hpu)
        elif device.isdigit():
            device = Device.hpu + ":" + device
    elif isinstance(device, torch.device):
        if str(device).startswith(Device.cuda):
            device = str(device).replace(Device.cuda, Device.hpu)
    elif isinstance(device, int):
        device = Device.hpu + ":" + str(device)
    return device


def kwargs_to_str(kwargs):
    ret = ""
    if isinstance(kwargs, dict):
        for key, val in kwargs.items():
            ret += "{}={}, ".format(key, val)
    else:
        return kwargs
    return ret


def get_device_index(device: Any) -> int:
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a HPU device. Note that for a HPU device without a specified index,
    i.e., ``torch.device('hpu')``, this will return the current default HPU
    device.

    If :attr:`device` is a Python integer, it is returned as is.
    """

    if isinstance(device, str):
        device = torch.device(device)
    device_idx: Optional[int] = None
    if isinstance(device, torch.device):
        device_idx = device.index
    if isinstance(device, int):
        device_idx = device
    if device_idx is None:
        device_idx = 0

    return device_idx
