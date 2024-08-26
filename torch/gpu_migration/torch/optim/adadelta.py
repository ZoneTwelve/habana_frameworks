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
from torch.optim import Adadelta
from typing import Optional

class Adadelta(Adadelta, TorchOptimizersModuleRegister):
    """
    .. py:gpumgrcall:: Adadelta.hpu_modified

    Sets foreach parameter to false.

    """

    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __init__(
        self,
        params,
        lr=1.0,
        rho=0.9,
        eps=1e-6,
        weight_decay=0,
        foreach: Optional[bool] = None,
        *,
        maximize: bool = False,
        differentiable: bool = False,
    ):
        foreach = False
        super().__init__(params, lr, rho, eps, weight_decay, foreach=foreach, maximize=maximize, differentiable=differentiable)