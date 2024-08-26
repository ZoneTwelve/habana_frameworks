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

from . import CudaModuleRegister
from habana_frameworks.torch.gpu_migration.core.register import BaseModuleRegister
from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER
from habana_frameworks.torch.gpu_migration.core import _utils
from typing import List, Union, Iterable
import torch
import habana_frameworks.torch.hpu.random as htrandom
import torch

from . import CudaModuleRegister


class CudaRandomModuleRegister(BaseModuleRegister):
    @classmethod
    def get_module(cls):
        if cls.enabled():
            #####################################
            # This piece of code is copied from gpu_migration/torch/cuda/__init__.py(141)device_count()
            # Because, the device_count() here returns 0
            try:
                import pyhlml  # type: ignore[import]
            except ModuleNotFoundError:
                raise ModuleNotFoundError("pyhlml module not found, please install pyhlml")
            pyhlml.hlmlInit()
            count = pyhlml.hlmlDeviceGetCount()
            pyhlml.hlmlShutdown()
            #####################################
            torch.cuda.default_generators = htrandom.default_generators * count
        return torch.cuda.random


@CudaRandomModuleRegister.register_f()
@CudaModuleRegister.register_f()
def get_rng_state(device: Union[int, str, torch.device] = "cuda") -> torch.Tensor:
    """
    .. py:gpumgrcall:: get_rng_state.hpu_match

    Maps torch.cuda.random.get_rng_state to torch.hpu.random.get_rng_state.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    device = _utils.convert_device_arg(device)
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call=f"torch.hpu.random.get_rng_state({device})",
    )
    return htrandom.get_rng_state(device)


@CudaRandomModuleRegister.register_f()
@CudaModuleRegister.register_f()
def get_rng_state_all() -> List[torch.Tensor]:
    """
    .. py:gpumgrcall:: get_rng_state_all.hpu_match

    Maps torch.cuda.random.get_rng_state_all to torch.hpu.random.get_rng_state_all.

    """
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        new_call=(
            "habana_frameworks.torch.gpu_migration.torch.hpu.random.get_rng_state_all()"
        ),
    )

    return htrandom.get_rng_state_all()


@CudaRandomModuleRegister.register_f()
@CudaModuleRegister.register_f()
def set_rng_state(
    new_state: torch.Tensor, device: Union[int, str, torch.device] = "cuda"
) -> None:
    """
    .. py:gpumgrcall:: set_rng_state.hpu_match

    Maps torch.cuda.random.set_rng_state to torch.hpu.random.set_rng_state.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    device = _utils.convert_device_arg(device)

    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.hpu.random.set_rng_state",
    )
    htrandom.set_rng_state(new_state, device)


@CudaRandomModuleRegister.register_f()
@CudaModuleRegister.register_f()
def set_rng_state_all(new_states: Iterable[torch.Tensor]) -> None:
    """
    .. py:gpumgrcall:: set_rng_state_all.hpu_match

    Maps torch.cuda.random.set_rng_state_all to torch.hpu.random.set_rng_state_all.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.hpu.random.set_rng_state_all",
    )
    htrandom.set_rng_state_all(new_states)


@CudaRandomModuleRegister.register_f()
@CudaModuleRegister.register_f()
def manual_seed(seed: int) -> None:
    """
    .. py:gpumgrcall:: manual_seed.hpu_match

    Maps torch.cuda.random.manual_seed to torch.hpu.random.manual_seed.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None

    htrandom.manual_seed(seed)

    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.hpu.random.manual_seed({})".format(seed),
    )


@CudaRandomModuleRegister.register_f()
@CudaModuleRegister.register_f()
def manual_seed_all(seed: int) -> None:
    """
    .. py:gpumgrcall:: manual_seed_all.hpu_match

    Maps torch.cuda.random.manual_seed_all to torch.hpu.random.manual_seed_all.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None

    htrandom.manual_seed_all(seed)

    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.hpu.random.manual_seed_all({})".format(seed),
    )


@CudaRandomModuleRegister.register_f()
@CudaModuleRegister.register_f()
def seed() -> None:
    """
    .. py:gpumgrcall:: seed.hpu_match

    Maps torch.cuda.random.seed to torch.hpu.random.seed.

    """
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        new_call="torch.hpu.random.seed()",
    )

    htrandom.seed()


@CudaRandomModuleRegister.register_f()
@CudaModuleRegister.register_f()
def seed_all() -> None:
    """
    .. py:gpumgrcall:: seed_all.hpu_match

    Maps torch.cuda.random.seed_all to torch.hpu.random.seed_all.

    """
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        new_call="torch.hpu.random.seed_all()",
    )

    htrandom.seed_all()


@CudaRandomModuleRegister.register_f()
@CudaModuleRegister.register_f()
def initial_seed() -> int:
    """
    .. py:gpumgrcall:: initial_seed.hpu_match

    Maps torch.cuda.random.initial_seed to torch.hpu.random.initial_seed.

    """
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        new_call="torch.hpu.random.initial_seed()",
    )

    return htrandom.initial_seed()
