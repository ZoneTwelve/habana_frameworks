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

import warnings

import torch

from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER
from habana_frameworks.torch.gpu_migration.torch import TORCH_VERSION

from . import CudaAmpModuleRegister

from typing import Dict, Any

class GradScaler(torch.cuda.amp.GradScaler, CudaAmpModuleRegister):
    """
    .. py:gpumgrcall:: hpu_modified

    Sets `enabled` argument to False. GradScaler prevents gradient values “underflow”, and used for ops with FP16 inputs. However, HPU uses BF16 in the training.

    """

    @classmethod
    def _save_orig_func_gpu_migration(cls):
        return ["__init__", "state_dict"]

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None

        if enabled:
            G_LOGGER.info(
                api_type="hpu_modified",
                func_prefix="torch.cuda.amp.GradScaler",
                old_args=log_args,
                new_call="set enabled to Flase",
            )
            enabled = False
            self.disabled_on_hpu = True
        else:
            self.disabled_on_hpu = False
        return GradScaler.call_parent_func(
            "__init__",
            self,
            init_scale,
            growth_factor,
            backoff_factor,
            growth_interval,
            enabled,
        )
    
    def state_dict(self) -> Dict[str, Any]:
        """
        .. py:gpumgrcall:: state_dict.hpu_match

        Returns the state of the scaler with values in disable mode.
        """
        warnings.warn(
            "GradScaler is not applicable to HPU. If this instance if disabled, the"
            " states of the scaler are values in disable mode."
        )
        if self.disabled_on_hpu:
            G_LOGGER.info(
                api_type="hpu_match",
                func_prefix="torch.cuda.amp.GradScaler",
                new_call="return state in diable mode",
            )
            return {
                "scale": self.get_scale(),
                "growth_factor": None,
                "backoff_factor": None,
                "growth_interval": None,
                "_growth_tracker": self._get_growth_tracker(),
            }
        else:
            return GradScaler.call_parent_func("state_dict", self)
