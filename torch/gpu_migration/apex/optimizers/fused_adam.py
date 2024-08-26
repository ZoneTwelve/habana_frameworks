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

import habana_frameworks.torch.hpex.optimizers.FusedAdamW as hpex_FusedAdamW
from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER
from torch.optim import Adam

from . import APEXOptimizersModuleRegister


class FusedAdam(APEXOptimizersModuleRegister):
    """
    .. py:gpumgrcall:: FusedAdam.hpu_modified

    Uses hpex_FusedAdamW if adam_w_mode is True. Otherwise, it uses torch.otpim.Adam.

    """

    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __new__(
        self,
        params,
        lr=1e-3,
        bias_correction=True,
        betas=(0.9, 0.999),
        eps=1e-8,
        adam_w_mode=True,
        weight_decay=0.0,
        amsgrad=False,
        set_grad_none=True,
    ):
        self.set_grad_none = set_grad_none
        if adam_w_mode:
            log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
            G_LOGGER.info(api_type="hpu_modified", func_prefix="apex.optimizers", old_args=log_args,
                        new_call="hpex.optimizers.FusedAdam(params, lr={}, betas={}, eps={}, weight_decay={})".format(lr, betas, eps, weight_decay))
            return hpex_FusedAdamW(
                params=params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
            )
        else:
            log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
            G_LOGGER.info(api_type="hpu_modified", func_prefix="apex.optimizers", old_args=log_args,
                        new_call="torch.optim.Adam(params, lr={}, betas={}, eps={}, weight_decay={}, amsgrad={})".format(lr, betas, eps, weight_decay, amsgrad))
            return Adam(
                params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
            )

    def zero_grad(self):
        """
        .. py:gpumgrcall:: zero_grad.hpu_match

        Maps apex.optimizers.FusedAdam.zero_grad to hpex.optimizers.FusedAdamW.zero_grad if adam_w_mode is set to True, otherwise maps to torch.optim.Adam.zero_grad.

        """
        if self.adam_w_mode:
            G_LOGGER.info(api_type="hpu_match", func_prefix="apex.optimizers",
                        new_call="hpex.optimizers.FusedAdam.zero_grad(set_to_none={})".format(self.set_to_none))
            hpex_FusedAdamW.zero_grad(self.set_grad_none)
        else:
            G_LOGGER.info(api_type="hpu_match", func_prefix="apex.optimizers",
                        new_call="torch.optim.Adam.zero_grad(set_to_none={})".format(self.set_to_none))
            Adam.zero_grad(self.set_grad_none)
