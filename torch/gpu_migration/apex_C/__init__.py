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

"""
The module to be replaced:
    - apex_C
"""
from typing import List

import torch


def unflatten(arg0: torch.Tensor, arg1: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    .. py:gpumgrcall:: unflatten.hpu_modified

        Uses torch._C._nn.unflatten_dense_tensors.
    """
    return torch._C._nn.unflatten_dense_tensors(arg0, arg1)


def flatten(arg0: List[torch.Tensor]) -> torch.Tensor:
    """
    .. py:gpumgrcall:: flatten.hpu_modified

        Uses torch._C._nn.flatten_dense_tensors.
    """
    return torch._C._nn.flatten_dense_tensors(arg0)
