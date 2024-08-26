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


from habana_frameworks.torch.gpu_migration.core.register import BaseModuleRegister
from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER

from . import CudaModuleRegister

class CudaCommModuleRegister(CudaModuleRegister):
    @classmethod
    def get_module(cls):
        import torch.cuda.comm

        return torch.cuda.comm


@CudaCommModuleRegister.register_f()
def broadcast(tensor, devices=None, *, out=None):
    """
    .. py:gpumgrcall:: broadcast.hpu_mismatch

    Raises NotImplementedError. Please use torch.distributed.broadcast instead.
    """
    G_LOGGER.info(
        api_type="hpu_mismatch",
        func_prefix="torch.cuda",
        new_call="raise NotImplementedError",
    )

    raise NotImplementedError("torch.cuda.comm.broadcast is currently not supported. Please use torch.distributed.broadcast instead.")


@CudaCommModuleRegister.register_f()
def broadcast_coalesced(tensors, devices, buffer_size=10485760):
    """
    .. py:gpumgrcall:: broadcast_coalesced.hpu_mismatch

    Raises NotImplementedError. Please use torch.distributed.broadcast instead.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    G_LOGGER.info(
        api_type="hpu_mismatch",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="raise NotImplementedError",
    )
    raise NotImplementedError("torch.cuda.comm.broadcast_coalesced is currently not supported. Please use torch.distributed.broadcast instead.")


@CudaCommModuleRegister.register_f()
def reduce_add(inputs, destination=None):
    """
    .. py:gpumgrcall:: reduce_add.hpu_mismatch

    Raises NotImplementedError. Please use torch.distributed.reduce instead.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    G_LOGGER.info(
        api_type="hpu_mismatch",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="raise NotImplementedError",
    )
    raise NotImplementedError("torch.cuda.comm.reduce_add is currently not supported. Please use torch.distributed.reduce instead.")


@CudaCommModuleRegister.register_f()
def reduce_add_coalesced(inputs, destination=None, buffer_size=10485760):
    """
    .. py:gpumgrcall:: reduce_add_coalesced.hpu_mismatch

    Raises NotImplementedError. Please use torch.distributed.reduce instead.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    G_LOGGER.info(
        api_type="hpu_mismatch",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="raise NotImplementedError",
    )
    raise NotImplementedError("torch.cuda.comm.reduce_add_coalesced is currently not supported. Please use torch.distributed.reduce instead.")


@CudaCommModuleRegister.register_f()
def scatter(tensor, devices=None, chunk_sizes=None, dim=0, streams=None, *, out=None):
    """
    .. py:gpumgrcall:: scatter.hpu_mismatch

    Raises NotImplementedError. Please use torch.distributed.scatter instead.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    G_LOGGER.info(
        api_type="hpu_mismatch",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="raise NotImplementedError",
    )
    raise NotImplementedError("torch.cuda.comm.scatter is currently not supported. Please use torch.distributed.scatter instead.")


@CudaCommModuleRegister.register_f()
def gather(tensors, dim=0, destination=None, *, out=None):
    """
    .. py:gpumgrcall:: gather.hpu_mismatch

    Raises NotImplementedError. Please use torch.distributed.gather instead.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    G_LOGGER.info(
        api_type="hpu_mismatch",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="raise NotImplementedError",
    )
    raise NotImplementedError("torch.cuda.comm.gather is currently not supported. Please use torch.distributed.gather instead.")
