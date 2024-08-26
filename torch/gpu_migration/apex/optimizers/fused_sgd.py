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

import habana_frameworks.torch.hpex.optimizers.FusedSGD as hpex_FusedSGD
from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER
from torch.optim.optimizer import required

from . import APEXOptimizersModuleRegister


class FusedSGD(hpex_FusedSGD, APEXOptimizersModuleRegister):
    """
    .. py:gpumgrcall:: FusedSGD.hpu_modified

    Ignores wd_after_momentum and materialize_master_grads.

    """

    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        wd_after_momentum=False,
        materialize_master_grads=True,
        set_grad_none=False,
    ):
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
        G_LOGGER.info(api_type="hpu_modified", func_prefix="apex.optimizers", old_args=log_args,
                    new_call="hpex.optimizers.FusedSGD(params, lr={}, momentum={}, dampening={}, \
                        weight_decay={}, nesterov={})".format(lr, momentum, dampening, weight_decay, nesterov))
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.set_grad_none = set_grad_none

    def zero_grad(self, set_to_none: bool = False):
        """
        .. py:gpumgrcall:: zero_grad.hpu_match

        Maps apex.optimizers.FusedSGD.zero_grad to hpex.optimizers.FusedSGD.zero_grad.

        """
        G_LOGGER.info(api_type="hpu_match", func_prefix="apex.optimizers",
                    new_call="hpex.optimizers.FusedSGD.zero_grad(set_to_none={})".format(self.set_to_none))
        FusedSGD.zero_grad(self, self.set_grad_none)
