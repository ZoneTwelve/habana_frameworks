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
from torch.optim import Adam
from typing import Optional

class Adam(Adam, TorchOptimizersModuleRegister):
    """
    .. py:gpumgrcall:: Adam.hpu_modified

    Sets foreach parameter to false.

    """

    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, *, foreach: Optional[bool] = None,
                 maximize: bool = False, capturable: bool = False,
                 differentiable: bool = False, fused: Optional[bool] = None):
        foreach = False
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, foreach=foreach, maximize=maximize, capturable=capturable, differentiable=differentiable, fused=fused)

