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

from . import TorchOptimizersModuleRegister
from torch.optim import RMSprop
from typing import Optional

class RMSprop(RMSprop, TorchOptimizersModuleRegister):
    """
    .. py:gpumgrcall:: RMSProp.hpu_modified

    Sets foreach parameter to false.

    """

    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
    ):
        foreach = False
        super().__init__(params, lr, alpha, eps, weight_decay, momentum, centered, foreach, maximize, differentiable)