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
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import apex
import habana_frameworks.torch.core as htcore
import importlib
import numbers

import torch
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.optim.optimizer import required

import habana_frameworks

from . import APEXNormFuncModuleRegister, APEXNormModuleRegister


class FusedLayerNorm(torch.nn.LayerNorm, APEXNormModuleRegister):
    """
    .. py:gpumgrcall:: FusedLayerNorm.hpu_modified

    Uses torch.nn.functional.layer_norm.

    """
    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

class MixedFusedLayerNorm(torch.nn.LayerNorm, APEXNormModuleRegister):
    """
    .. py:gpumgrcall:: MixedFusedLayerNorm.hpu_modified

    Uses torch.nn.functional.layer_norm.

    """
    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

class FusedLayerNormAffineFunction(torch.autograd.Function, APEXNormFuncModuleRegister):
    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def apply(input, weight, bias, normalized_shape, eps):
        """
        .. py:gpumgrcall:: FusedLayerNormAffineFunction.hpu_modified

        Uses torch.nn.functional.layer_norm.

        """
        return torch.nn.functional.layer_norm(
            input, normalized_shape, weight, bias, eps
        )


class FusedLayerNormFunction(torch.autograd.Function, APEXNormFuncModuleRegister):
    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def apply(input, weight, bias, normalized_shape, eps):
        """
        .. py:gpumgrcall:: FusedLayerNormFunction.hpu_modified

        Uses to torch.nn.functional.layer_norm.

        """
        return torch.nn.functional.layer_norm(input, normalized_shape, None, None, eps)

# Reference implementation from Huggingface
def manual_rms_norm(input, normalized_shape, weight, eps):
    # layer norm should always be calculated in float32
    dims = tuple(i for i in range(-1, -len(normalized_shape)-1, -1))
    variance = input.to(torch.float32).pow(2).mean(dims, keepdim=True)
    input = input * torch.rsqrt(variance + eps)

    if weight is None:
        return input

    # convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(weight.dtype)

    return weight * input

class FusedRMSNorm(torch.nn.Module, APEXNormModuleRegister):
    """
    .. py:gpumgrcall:: FusedRMSNorm.hpu_modified

    Uses python manual RMSNorm implementation; ignores elementwise_affine option.

    """

    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(*normalized_shape))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, input):
        return manual_rms_norm(input, self.normalized_shape, self.weight, self.eps)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(**self.__dict__)

    
class MixedFusedRMSNorm(FusedRMSNorm, APEXNormModuleRegister):
    """
    .. py:gpumgrcall:: MixedFusedRMSNorm.hpu_modified

    Maps to FusedRMSNorm.

    """
    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)