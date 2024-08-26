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

from . import APEXParallelModuleRegister


class DistributedDataParallel(APEXParallelModuleRegister):
    """
    .. py:gpumgrcall:: hpu_mismatch

    Raises NotImplementedError. Please use native torch.nn.DistributedDataParallel.


    """

    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __init__(
        self,
        module,
        message_size=10000000,
        delay_allreduce=False,
        shared_param=None,
        allreduce_trigger_params=None,
        retain_allreduce_buffers=False,
        allreduce_always_fp32=False,
        num_allreduce_streams=1,
        allreduce_communicators=None,
        gradient_average=True,
        gradient_predivide_factor=1.0,
        gradient_average_split_factor=None,
        prof=False,
    ):
        raise NotImplementedError(
            "Apex parallel DistributedDataParallel is not supported. Please use native torch.nn.DistributedDataParallel."
        )


class Reducer(APEXParallelModuleRegister):
    """
    .. py:gpumgrcall:: hpu_mismatch

    Raises NotImplementedError. Please use torch.distributed.reduce instead.

    """

    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __init__(module_or_grads_list):
        raise NotImplementedError("Apex parallel Reducer is not supported. Please use torch.distributed.reduce instead.")


class SyncBatchNorm(APEXParallelModuleRegister):
    """
    .. py:gpumgrcall:: hpu_mismatch

    Raises NotImplementedError. Please use torch.nn.SyncBatchNorm instead.

    """

    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        process_group=None,
        channel_last=False,
        fuse_relu=False,
    ):
        raise NotImplementedError("Apex parallel SyncBatchNorm is not supported. Please use torch.nn.SyncBatchNorm instead.")


@APEXParallelModuleRegister.register_f()
def convert_syncbn_model(module, process_group=None, channel_last=False):
    """
    .. py:gpumgrcall:: convert_syncbn_model.hpu_mismatch

    Raises NotImplementedError. Please convert torch.nn.modules.batchnorm._BatchNorm to torch.nn.SyncBatchNorm manually.

    """
    raise NotImplementedError("Apex parallel convert_syncbn_model is not supported. Please convert torch.nn.modules.batchnorm._BatchNorm to torch.nn.SyncBatchNorm manually.")
