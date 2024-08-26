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

import habana_frameworks.torch.hpex.optimizers.FusedAdagrad as hpex_FusedAdagrad
from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER

from . import APEXOptimizersModuleRegister


class FusedAdagrad(hpex_FusedAdagrad, APEXOptimizersModuleRegister):
    """
    .. py:gpumgrcall:: FusedAdagrad.hpu_modified

    Does not support adagrad_w_mode.

    """

    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __init__(
        self,
        params,
        lr=0.01,
        eps=1e-10,
        weight_decay=0.0,
        set_grad_none=True,
        adagrad_w_mode=False,
    ):
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
        G_LOGGER.info(api_type="hpu_modified", func_prefix="apex.optimizers", old_args=log_args,
                    new_call="hpex.optimizers.FusedAdagrad(params, lr={}, weight_decay={}, eps={})".format(lr, weight_decay, eps))
        if adagrad_w_mode:
            raise ImportError("adagrad_w_mode is not supported")
        hpex_FusedAdagrad.__init__(self, params, lr, weight_decay, eps)
        self.set_grad_none = set_grad_none

    def zero_grad(self, set_to_none: bool = False):
        """
        .. py:gpumgrcall:: zero_grad.hpu_match

        Maps apex.optimizers.FusedAdagrad.zero_grad to hpex.optimizers.FusedAdagrad.zero_grad.

        """
        G_LOGGER.info(api_type="hpu_match", func_prefix="apex.optimizers",
                    new_call="hpex.optimizers.FusedAdagrad.zero_grad(set_to_none={})".format(set_to_none))
        hpex_FusedAdagrad.zero_grad(self, self.set_grad_none)
