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

import habana_frameworks.torch.hpex.optimizers.FusedLamb as hpex_FusedLamb
from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER

from . import APEXOptimizersModuleRegister


class FusedLAMB(hpex_FusedLamb, APEXOptimizersModuleRegister):
    """
    .. py:gpumgrcall:: FusedLAMB.hpu_match

    Maps apex.optimizers.FusedLamb to hpex.optimizers.FusedLamb.

    """

    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __init__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.01,
        amsgrad=False,
        adam_w_mode=True,
        grad_averaging=True,
        set_grad_none=True,
        max_grad_norm=1.0,
        use_nvlamb=False,
    ):

        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
        G_LOGGER.info(api_type="hpu_match", func_prefix="apex.optimizers", old_args=log_args,
                    new_call="hpex.optimizers.FusedLAMB(params, lr={}, bias_correction={}, \
                        betas={}, eps={}, weight_decay={}, amsgrad={}, adam_w_mode={} \
                        grad_averaging={}, set_grad_none={}, max_grad_norm={}, use_nvlamb={} \)"
                        .format(lr, bias_correction, betas, eps, weight_decay, amsgrad, adam_w_mode,
                                grad_averaging, set_grad_none, max_grad_norm, use_nvlamb))
        hpex_FusedLamb.__init__(
            self,
            params,
            lr,
            bias_correction,
            betas,
            eps,
            weight_decay,
            amsgrad,
            adam_w_mode,
            grad_averaging,
            set_grad_none,
            max_grad_norm,
            use_nvlamb,
        )
        self.set_grad_none = set_grad_none

    def zero_grad(self, set_to_none: bool = False):
        """
        .. py:gpumgrcall:: zero_grad.hpu_match

        Maps apex.optimizers.FusedLamb.zero_grad to hpex.optimizers.FusedLamb.zero_grad.

        """
        G_LOGGER.info(api_type="hpu_match", func_prefix="apex.optimizers",
                    new_call="hpex.optimizers.FusedLamb.zero_grad(set_to_none={})".format(self.set_to_none))
        hpex_FusedLamb.zero_grad(self, self.set_grad_none)
