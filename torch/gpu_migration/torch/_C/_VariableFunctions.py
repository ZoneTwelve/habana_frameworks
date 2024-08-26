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
import warnings

from habana_frameworks.torch.gpu_migration.core._utils import (
    convert_device_arg, kwargs_to_str)
from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER
from habana_frameworks.torch.gpu_migration.torch import TorchModuleRegister

CONVERT_FP16_TO_BF16 = os.getenv("PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION", "0") == "1"

__all__ = [
    "empty",
    "empty_like",
    "randn",
    "rand_like",
    "rand",
    "randint",
    "randint_like",
    "randn_like",
    "tensor",
    "zeros",
    "zeros_like",
    "arange",
    "range",
    "full",
    "full_like",
    "eye",
    "empty_strided",
    "ones",
    "ones_like",
    "as_tensor",
    "randperm"
]


def update_params(func_name, *args, **kwargs):
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    should_log = False
    api_type = "hpu_match"

    device = kwargs.get("device")
    if device is not None:
        kwargs['device'] = convert_device_arg(device)
        should_log = True if not device == kwargs['device'] else False
        dtype = kwargs.get("dtype")
        if dtype == torch.float16 and CONVERT_FP16_TO_BF16 and "hpu" in str(kwargs['device']):
            kwargs['dtype'] = torch.bfloat16
            should_log = True

    if 'pin_memory' in kwargs:
        pin_memory = kwargs.get("pin_memory")
        if pin_memory == True:
            kwargs['pin_memory'] = False
            should_log = True
            api_type = "hpu_modified"
            warnings.warn("habana allocator doesn't support pinned memory")

    if should_log and log_args is not None:
        del log_args['func_name']
        G_LOGGER.info(api_type=api_type, func_prefix="torch", old_args=log_args,
                      new_call="torch.{}(args={}, kwargs={{{}}})".format(func_name, args, kwargs_to_str(kwargs)), stack=2)

    return args, kwargs


@TorchModuleRegister.register_op()
def empty(*args, **kwargs):
    """
    .. py:gpumgrcall:: empty.hpu_modified

    Changes device arguments from "cuda" to "hpu" and dtype from torch.float16 to torch.bfloat16 if PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1. Ignores pin_memory option.

    """
    args, kwargs = update_params("empty", *args, **kwargs)
    return TorchModuleRegister.empty(*args, **kwargs)


@TorchModuleRegister.register_op()
def empty_like(*args, **kwargs):
    """
    .. py:gpumgrcall:: empty_like.hpu_match

    Changes device arguments from "cuda" to "hpu" and dtype from torch.float16 to torch.bfloat16 if PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1.

    """
    args, kwargs = update_params("empty_like", *args, **kwargs)
    return TorchModuleRegister.empty_like(*args, **kwargs)


@TorchModuleRegister.register_op()
def randn(*args, **kwargs):
    """
    .. py:gpumgrcall:: randn.hpu_modified

    Changes device arguments from "cuda" to "hpu" and dtype from torch.float16 to torch.bfloat16 if PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1. Ignores pin_memory option.

    """
    args, kwargs = update_params("randn", *args, **kwargs)
    return TorchModuleRegister.randn(*args, **kwargs)


@TorchModuleRegister.register_op()
def rand_like(*args, **kwargs):
    """
    .. py:gpumgrcall:: rand_like.hpu_match

    Changes device arguments from "cuda" to "hpu" and dtype from torch.float16 to torch.bfloat16 if PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1.

    """
    args, kwargs = update_params("rand_like", *args, **kwargs)
    return TorchModuleRegister.rand_like(*args, **kwargs)


@TorchModuleRegister.register_op()
def rand(*args, **kwargs):
    """
    .. py:gpumgrcall:: rand.hpu_modified

    Changes device arguments from "cuda" to "hpu" and dtype from torch.float16 to torch.bfloat16 if PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1. Ignores pin_memory option.

    """
    args, kwargs = update_params("rand", *args, **kwargs)
    return TorchModuleRegister.rand(*args, **kwargs)


@TorchModuleRegister.register_op()
def randint(*args, **kwargs):
    """
    .. py:gpumgrcall:: randint.hpu_match

    Changes device arguments from "cuda" to "hpu" and dtype from torch.float16 to torch.bfloat16 if PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1.

    """
    args, kwargs = update_params("randint", *args, **kwargs)
    return TorchModuleRegister.randint(*args, **kwargs)


@TorchModuleRegister.register_op()
def randint_like(*args, **kwargs):
    """
    .. py:gpumgrcall:: randint_like.hpu_match

    Changes device arguments from "cuda" to "hpu" and dtype from torch.float16 to torch.bfloat16 if PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1.

    """
    args, kwargs = update_params("randint_like", *args, **kwargs)
    return TorchModuleRegister.randint_like(*args, **kwargs)


@TorchModuleRegister.register_op()
def randn_like(*args, **kwargs):
    """
    .. py:gpumgrcall:: randn_like.hpu_match

    Changes device arguments from "cuda" to "hpu" and dtype from torch.float16 to torch.bfloat16 if PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1.

    """
    args, kwargs = update_params("randn_like", *args, **kwargs)
    return TorchModuleRegister.randn_like(*args, **kwargs)


@TorchModuleRegister.register_op()
def tensor(*args, **kwargs):
    """
    .. py:gpumgrcall:: tensor.hpu_match

    Changes device arguments from "cuda" to "hpu" and dtype from torch.float16 to torch.bfloat16 if PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1.

    """
    args, kwargs = update_params("tensor", *args, **kwargs)
    return TorchModuleRegister.tensor(*args, **kwargs)


@TorchModuleRegister.register_op()
def zeros(*args, **kwargs):
    """
    .. py:gpumgrcall:: zeros.hpu_match

    Changes device arguments from "cuda" to "hpu" and dtype from torch.float16 to torch.bfloat16 if PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1.

    """
    args, kwargs = update_params("zeros", *args, **kwargs)
    return TorchModuleRegister.zeros(*args, **kwargs)


@TorchModuleRegister.register_op()
def zeros_like(*args, **kwargs):
    """
    .. py:gpumgrcall:: zeros_like.hpu_match

    Changes device arguments from "cuda" to "hpu" and dtype from torch.float16 to torch.bfloat16 if PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1.

    """
    args, kwargs = update_params("zeros_like", *args, **kwargs)
    return TorchModuleRegister.zeros_like(*args, **kwargs)


@TorchModuleRegister.register_op()
def arange(*args, **kwargs):
    """
    .. py:gpumgrcall:: arange.hpu_match

    Changes device arguments from "cuda" to "hpu" and dtype from torch.float16 to torch.bfloat16 if PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1.

    """
    args, kwargs = update_params("arange", *args, **kwargs)
    return TorchModuleRegister.arange(*args, **kwargs)


@TorchModuleRegister.register_op()
def range(*args, **kwargs):
    """
    .. py:gpumgrcall:: range.hpu_match

    Changes device arguments from "cuda" to "hpu" and dtype from torch.float16 to torch.bfloat16 if PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1.

    """
    args, kwargs = update_params("range", *args, **kwargs)
    return TorchModuleRegister.range(*args, **kwargs)


@TorchModuleRegister.register_op()
def full(*args, **kwargs):
    """
    .. py:gpumgrcall:: full.hpu_match

    Changes device arguments from "cuda" to "hpu" and dtype from torch.float16 to torch.bfloat16 if PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1.

    """
    args, kwargs = update_params("full", *args, **kwargs)
    return TorchModuleRegister.full(*args, **kwargs)


@TorchModuleRegister.register_op()
def full_like(*args, **kwargs):
    """
    .. py:gpumgrcall:: full_like.hpu_match

    Changes device arguments from "cuda" to "hpu" and dtype from torch.float16 to torch.bfloat16 if PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1.

    """
    args, kwargs = update_params("full_like", *args, **kwargs)
    return TorchModuleRegister.full_like(*args, **kwargs)


@TorchModuleRegister.register_op()
def eye(*args, **kwargs):
    """
    .. py:gpumgrcall:: eye.hpu_match

    Changes device arguments from "cuda" to "hpu" and dtype from torch.float16 to torch.bfloat16 if PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1.

    """
    args, kwargs = update_params("eye", *args, **kwargs)
    return TorchModuleRegister.eye(*args, **kwargs)


@TorchModuleRegister.register_op()
def empty_strided(*args, **kwargs):
    """
    .. py:gpumgrcall:: empty_strided.hpu_modified

    Changes device arguments from "cuda" to "hpu" and dtype from torch.float16 to torch.bfloat16 if PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1. Ignores pin_memory option.

    """
    args, kwargs = update_params("empty_strided", *args, **kwargs)
    return TorchModuleRegister.empty_strided(*args, **kwargs)


@TorchModuleRegister.register_op()
def ones(*args, **kwargs):
    """
    .. py:gpumgrcall:: ones.hpu_match

    Changes device arguments from "cuda" to "hpu" and dtype from torch.float16 to torch.bfloat16 if PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1.

    """
    args, kwargs = update_params("ones", *args, **kwargs)
    return TorchModuleRegister.ones(*args, **kwargs)


@TorchModuleRegister.register_op()
def ones_like(*args, **kwargs):
    """
    .. py:gpumgrcall:: ones_like.hpu_match

    Changes device arguments from "cuda" to "hpu" and dtype from torch.float16 to torch.bfloat16 if PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1.

    """
    args, kwargs = update_params("ones_like", *args, **kwargs)
    return TorchModuleRegister.ones_like(*args, **kwargs)


@TorchModuleRegister.register_op()
def as_tensor(*args, **kwargs):
    """
    .. py:gpumgrcall:: as_tensor.hpu_match

    Changes device arguments from "cuda" to "hpu" and dtype from torch.float16 to torch.bfloat16 if PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1.

    """
    args, kwargs = update_params("as_tensor", *args, **kwargs)
    return TorchModuleRegister.as_tensor(*args, **kwargs)


@TorchModuleRegister.register_op()
def randperm(*args, **kwargs):
    """
    .. py:gpumgrcall:: randperm.hpu_modified

    Changes device arguments from "cuda" to "hpu" and dtype from torch.float16 to torch.bfloat16 if PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION=1. Ignores pin_memory option.

    """
    args, kwargs = update_params("randperm", *args, **kwargs)
    return TorchModuleRegister.randperm(*args, **kwargs)
