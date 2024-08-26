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

import os
import subprocess
import warnings
from typing import Any, List, Optional, Tuple, Union

import torch
from habana_frameworks.torch.hpu import (HABANA_VISIBLE_MODULES_VAR,
                                         HLS_MODULE_ID_VAR)

from habana_frameworks.torch.gpu_migration.core import _utils
from habana_frameworks.torch.gpu_migration.core._enums import Device
from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER
from habana_frameworks.torch.gpu_migration.core.register import \
    BaseModuleRegister
from habana_frameworks.torch.gpu_migration.torch import TORCH_VERSION

class CudaModuleRegister(BaseModuleRegister):
    @classmethod
    def get_module(cls):
        return torch.cuda


from . import amp, comm, graphs, memory, nccl, nvtx, random, streams


class device(torch.hpu.device, CudaModuleRegister):
    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)


class device_of(torch.hpu.device_of, CudaModuleRegister):
    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)


class DeferredCudaCallError(Exception, CudaModuleRegister):
    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)


@CudaModuleRegister.register_f()
def init():
    """
    .. py:gpumgrcall:: init.hpu_match

    Maps torch.cuda.init to torch.hpu.init.
    """
    G_LOGGER.info(
        api_type="hpu_match", func_prefix="torch.cuda", new_call="torch.hpu.init()"
    )

    torch.hpu.init()


@CudaModuleRegister.register_f()
def cudart():
    """
    .. py:gpumgrcall:: cudart.hpu_modified

    Returns None.

    """
    G_LOGGER.info(
        api_type="hpu_modified", func_prefix="torch.cuda", new_call="return None"
    )

    return None


class CudaError(RuntimeError, CudaModuleRegister):
    """
    .. py:gpumgrcall:: hpu_modified

    Uses RuntimeError.

    """

    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)


@CudaModuleRegister.register_f()
def check_error(res: int) -> None:
    """
    .. py:gpumgrcall:: check_error.hpu_modified

    Uses RuntimeError.

    """
    if res != 0:
        raise torch.cuda.CudaError(res)


@CudaModuleRegister.register_f()
def is_initialized():
    """
    .. py:gpumgrcall:: is_initialized.hpu_match

    Maps torch.cuda.is_initialized to torch.hpu.is_initialized.
    """
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        new_call="torch.hpu.is_initialized()",
    )

    return torch.hpu.is_initialized()


@CudaModuleRegister.register_f()
def is_available() -> bool:
    """
    .. py:gpumgrcall:: is_available.hpu_match

    Maps torch.cuda.is_available to torch.hpu.is_available.
    """
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        new_call="torch.hpu.is_available()",
    )

    return torch.hpu.is_available()


@CudaModuleRegister.register_f()
def device_count() -> int:
    """
    .. py:gpumgrcall:: device_count.hpu_modified

    Uses pyhlml to get the device count.

    """
    try:
        import pyhlml  # type: ignore[import]
    except ModuleNotFoundError:
        raise ModuleNotFoundError("pyhlml module not found, please install pyhlml")
    pyhlml.hlmlInit()
    count = pyhlml.hlmlDeviceGetCount()
    pyhlml.hlmlShutdown()

    G_LOGGER.info(
        api_type="hpu_modified",
        func_prefix="torch.cuda",
        new_call="habana_frameworks.torch.gpu_migration.torch.cuda.device_count()",
    )
    return count


@CudaModuleRegister.register_f()
def get_device_name(device: Optional[torch.cuda._device_t] = None) -> str:
    """
    .. py:gpumgrcall:: get_device_name.hpu_match

    Maps torch.cuda.get_device_name to torch.hpu.get_device_name.

    """
    if device is not None:
        device = _utils.convert_device_arg(device)
    return torch.hpu.get_device_name(device)


@CudaModuleRegister.register_f()
def current_device() -> int:
    """
    .. py:gpumgrcall:: current_device.hpu_modified

    Uses HABANA_VISIBLE_MODULES_VAR and HLS_MODULE_ID_VAR to get the device ID.

    """
    G_LOGGER.info(
        api_type="hpu_modified",
        func_prefix="torch.cuda",
        new_call="habana_frameworks.torch.gpu_migration.torch.cuda.current_device()",
    )

    if (
        HABANA_VISIBLE_MODULES_VAR in os.environ.keys()
        and HLS_MODULE_ID_VAR in os.environ.keys()
    ):
        # the case using specific cards
        visible_modules = os.environ[HABANA_VISIBLE_MODULES_VAR].split(",")
        for idx, mod in enumerate(visible_modules):
            if mod == os.environ[HLS_MODULE_ID_VAR]:
                return idx
        raise ValueError(
            "Cannot find module_id {} in visible_module_ids {}".format(
                HLS_MODULE_ID_VAR, HABANA_VISIBLE_MODULES_VAR
            )
        )
    elif "LOCAL_RANK" in os.environ.keys():
        # multicard env variable which set by torchrun and overrided by habana pytorch bridge
        return int(os.environ["LOCAL_RANK"])
    elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ.keys():
        # multicard env variable which set by mpirun
        return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    else:
        # single card and multicard both may run into here, but don't know how to get the current device in multicard case
        curr_device = torch.hpu.current_device()
        return curr_device


@CudaModuleRegister.register_f()
def synchronize(device: torch.cuda._device_t = None) -> None:
    """
    .. py:gpumgrcall:: synchronize.hpu_match

    Maps torch.cuda.synchronize to torch.hpu.synchronize.
    """
    if device is not None:
        device = _utils.convert_device_arg(device)
    torch.hpu.synchronize()

    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        new_call="torch.hpu.synchronize()",
    )


@CudaModuleRegister.register_f()
def ipc_collect():
    """
    .. py:gpumgrcall:: ipc_collect.hpu_modified

    Returns None.

    """
    G_LOGGER.info(api_type="hpu_modified", func_prefix="torch.cuda", new_call="Dummy")
    return None


@CudaModuleRegister.register_f()
def current_stream(
    device: Optional[torch.cuda._device_t] = None,
) -> torch.cuda.streams.Stream:
    """
    .. py:gpumgrcall:: current_stream.hpu_match

    Maps torch.cuda.current_stream to torch.hpu.current_stream.
    """
    if device is not None:
        device = _utils.convert_device_arg(device)

    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        new_call="torch.hpu.current_stream",
    )

    return torch.hpu.current_stream()


@CudaModuleRegister.register_f()
def default_stream(
    device: Optional[torch.cuda._device_t] = None,
) -> torch.cuda.streams.Stream:
    """
    .. py:gpumgrcall:: default_stream.hpu_match

    Maps torch.cuda.default_stream to torch.hpu.default_stream.
    """
    if device is not None:
        device = _utils.convert_device_arg(device)

    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        new_call="torch.hpu.default_stream",
    )

    return torch.hpu.default_stream()


@CudaModuleRegister.register_f()
def current_blas_handle():
    """
    .. py:gpumgrcall:: current_blas_handle.hpu_modified

    Returns 0.

    """
    G_LOGGER.info(
        api_type="hpu_modified", func_prefix="torch.cuda", new_call="return 0"
    )
    return 0


@CudaModuleRegister.register_f()
def set_sync_debug_mode(debug_mode: Union[int, str]) -> None:
    """
    .. py:gpumgrcall:: set_sync_debug_mode.hpu_match

    Maps torch.cuda.set_sync_debug_mode to torch.hpu.set_sync_debug_mode.
    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.hpu.set_sync_debug_mode({})".format(debug_mode),
    )

    torch.hpu.set_sync_debug_mode(debug_mode)


@CudaModuleRegister.register_f()
def get_sync_debug_mode() -> int:
    """
    .. py:gpumgrcall:: get_sync_debug_mode.hpu_modified

    Uses PT_ENABLE_HABANA_STREAMASYNC environment variable to get get_sync_debug_mode.

    """
    # TODO: [SW-107268] Implement torch.hpu.get_sync_debug_mode
    G_LOGGER.info(
        api_type="hpu_modified",
        func_prefix="torch.cuda",
        new_call=(
            "habana_frameworks.torch.gpu_migration.torch.cuda.get_sync_debug_mode()"
        ),
    )

    import os

    return int(os.environ["PT_ENABLE_HABANA_STREAMASYNC"])


# remove this when SW-107172 is Done
def _get_device_index(
    device: Any, optional: bool = False, allow_cpu: bool = False
) -> int:
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a HPU device. Note that for a HPU device without a specified index,
    i.e., ``torch.device('hpu')``, this will return the current default HPU
    device.

    If :attr:`device` is a Python integer, it is returned as is.
    """

    if isinstance(device, str):
        device = torch.device(device)
    device_idx: Optional[int] = None
    if isinstance(device, torch.device):
        if allow_cpu:
            if device.type not in ["hpu", "cpu"]:
                raise ValueError(
                    "Expected a cuda or cpu device, but got: {}".format(device)
                )
        elif device.type != "hpu":
            raise ValueError("Expected a hpu device, but got: {}".format(device))
        device_idx = -1 if device.type == "cpu" else device.index
    if isinstance(device, int):
        device_idx = device
    if device_idx is None:
        device_idx = torch.hpu.current_device()

    return device_idx


@CudaModuleRegister.register_f()
def memory_usage(device: Optional[Union[torch.types.Device, int]] = None) -> int:
    """
    .. py:gpumgrcall:: memory_usage.hpu_match

    Maps to torch.hpu.memory_usage().

    """
    device = _utils.convert_device_arg(device)

    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call=(
            "habana_frameworks.torch.gpu_migration.torch.cuda.memory_usage(device={})"
            .format(device)
        ),
    )

    return torch.hpu.memory_usage()

@CudaModuleRegister.register_f()
def utilization(device: Optional[Union[torch.types.Device, int]] = None) -> int:
    """
    .. py:gpumgrcall:: utilization.hpu_match

    Maps to torch.hpu.utilization().

    """
    device = _utils.convert_device_arg(device)

    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call=(
            "habana_frameworks.torch.gpu_migration.torch.cuda.utilization(device={})"
            .format(device)
        ),
    )
    return torch.hpu.utilization()

# TODO: Please remove else path once PT2.4 will be the default version
if TORCH_VERSION.startswith('2.4'):
    @CudaModuleRegister.register_f()
    def is_bf16_supported(including_emulation: bool = True):
        """
        .. py:gpumgrcall:: is_bf16_supported.hpu_match

        Maps torch.cuda.is_bf16_supported to torch.hpu.is_bf16_supported.

        """
        G_LOGGER.info(
            api_type="hpu_match",
            func_prefix="torch.cuda",
            new_call="torch.hpu.is_bf16_supported()",
        )
        return torch.hpu.is_bf16_supported()
else:
    @CudaModuleRegister.register_f()
    def is_bf16_supported():
        """
        .. py:gpumgrcall:: is_bf16_supported.hpu_match

        Maps torch.cuda.is_bf16_supported to torch.hpu.is_bf16_supported.

        """
        G_LOGGER.info(
            api_type="hpu_match",
            func_prefix="torch.cuda",
            new_call="torch.hpu.is_bf16_supported()",
        )
        return torch.hpu.is_bf16_supported()


@CudaModuleRegister.register_f()
def _sleep(cycles):
    r"""About 1398753 cycles per sec; measured on A100"""
    import time

    time.sleep(cycles / 1398753.0 / 1000.0)


@CudaModuleRegister.register_f()
def get_device_capability(
    device: Optional[torch.cuda._device_t] = None,
) -> Tuple[int, int]:
    """
    .. py:gpumgrcall:: get_device_capability.hpu_modified

    Returns latest cuda capability.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    G_LOGGER.info(
        api_type="hpu_modified",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="return (9, 0)",
    )

    return (9, 0)


# algin to torch.cuda._CudaDeviceProperties
class _CudaDeviceProperties:
    def __init__(
        self,
        name: str,
        major: int,
        minor: int,
        multi_processor_count: int,
        total_memory: int,
    ):
        self.name = name
        self.major = major
        self.minor = minor
        self.multi_processor_count = multi_processor_count
        self.total_memory = total_memory
        # self.is_integrated: int
        # self.is_multi_gpu_board: int


@CudaModuleRegister.register_f()
def get_device_properties(
    device: torch.cuda._device_t,
) -> torch.cuda._CudaDeviceProperties:
    """
    .. py:gpumgrcall:: get_device_properties.hpu_modified

    Returns a constructed _CudaDeviceProperties.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None

    device = _utils.convert_device_arg(device)

    G_LOGGER.info(
        api_type="hpu_modified",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.cuda.get_device_properties(device={})".format(device),
    )

    import multiprocessing

    cpu = multiprocessing.cpu_count()
    from habana_frameworks.torch import _hpu_C

    if not torch.hpu.is_available():
        warnings.warn("Device not available")
        return ""

    torch.cuda.init()
    device = torch.hpu._get_device_index(device, optional=True)
    if device < 0 or device >= torch.cuda.device_count():
        raise AssertionError("Invalid device id")
    mem = _hpu_C.get_mem_stats(device)["Limit"]
    name = torch.cuda.get_device_name(device)
    major, minor = torch.cuda.get_device_capability()
    return _CudaDeviceProperties(
        name=name, major=major, minor=minor, total_memory=mem, multi_processor_count=cpu
    )


class StreamContext(torch.hpu.StreamContext, CudaModuleRegister):
    """
    .. py:gpumgrcall:: hpu_match

    Maps torch.cuda.StreamContext to torch.hpu.StreamContext.
    """

    is_replace_class = True

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)


@CudaModuleRegister.register_f()
def set_stream(stream: torch.cuda.streams.Stream):
    """
    .. py:gpumgrcall:: set_stream.hpu_match

    Maps torch.cuda.set_stream to torch.hpu.set_stream.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.hpu.set_stream({})".format(stream),
    )

    return torch.hpu.set_stream(stream)


@CudaModuleRegister.register_f()
def stream(stream: Optional["torch.cuda.Stream"]) -> torch.cuda.StreamContext:
    """
    .. py:gpumgrcall:: stream.hpu_modified

    Returns a torch.cuda.StreamContext.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.hpu.stream({})".format(
            stream
        ),
    )

    return torch.hpu.stream(stream)


@CudaModuleRegister.register_f()
def can_device_access_peer(
    device: torch.cuda._device_t, peer_device: torch.cuda._device_t
) -> bool:
    """
    .. py:gpumgrcall:: can_device_access_peer.hpu_match

    Maps torch.cuda.can_device_access_peer to torch.hpu.can_device_access_peer.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None

    device = _utils.convert_device_arg(device)
    peer_device = _utils.convert_device_arg(peer_device)

    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.hpu.can_device_access_peer({}, {})".format(device, peer_device),
    )

    return torch.hpu.can_device_access_peer(device, peer_device)


@CudaModuleRegister.register_f()
def get_gencode_flags() -> str:
    """
    .. py:gpumgrcall:: get_gencode_flags.hpu_match

    Maps torch.cuda.get_gencode_flags to torch.hpu.get_gencode_flags.

    """
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        new_call="torch.hpu.get_gencode_flags()",
    )
    return torch.hpu.get_gencode_flags()


@CudaModuleRegister.register_f()
def get_arch_list() -> List[str]:
    """
    .. py:gpumgrcall:: get_arch_list.hpu_modified

    Returns a constructed list of all nvcc archs.

    """
    # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-name-gpuname-arch
    G_LOGGER.info(
        api_type="hpu_modified",
        func_prefix="torch.cuda",
        new_call="habana_frameworks.torch.gpu_migration.torch.hpu.get_arch_list()",
    )
    return [
        "compute_50",
        "compute_52",
        "compute_53",
        "compute_60",
        "compute_61",
        "compute_62",
        "compute_70",
        "compute_72",
        "compute_75",
        "compute_80",
        "compute_86",
        "compute_87",
        "compute_89",
        "compute_90",
        "lto_50",
        "lto_52",
        "lto_53",
        "lto_60",
        "lto_61",
        "lto_62",
        "lto_70",
        "lto_72",
        "lto_75",
        "lto_80",
        "lto_86",
        "lto_87",
        "lto_89",
        "lto_90",
        "sm_50",
        "sm_52",
        "sm_53",
        "sm_60",
        "sm_61",
        "sm_62",
        "sm_70",
        "sm_72",
        "sm_75",
        "sm_80",
        "sm_86",
        "sm_87",
        "sm_89",
        "sm_90",
    ]


@CudaModuleRegister.register_f()
def set_device(device: torch.cuda._device_t) -> None:
    """
    .. py:gpumgrcall:: set_device.hpu_match

    Maps to torch.hpu.set_device.

    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None

    device = _utils.convert_device_arg(device)
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.hpu.set_device({})".format(device),
    )

    torch.hpu.set_device(device)


class FloatTensor(CudaModuleRegister):
    """
    .. py:gpumgrcall:: hpu_modified

    Maps torch.cuda.FloatTensor to torch.FloatTensor + torch.Tensor.to(“hpu”). torch.cuda.FloatTensor is no longer the tensor type, and can only be used to create a tensor on HPU.

    """

    is_replace_class = True
    tensor_type = torch.float32

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __new__(self, *args, **kwargs):
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
        G_LOGGER.info(
            api_type="hpu_modified",
            func_prefix="torch.cuda",
            old_args=log_args,
            new_call="torch.FloatTensor(args={}, kwargs={{{}}}).to({})".format(
                args, _utils.kwargs_to_str(kwargs), Device.hpu
            ),
        )

        return torch.FloatTensor(*args, **kwargs).to(Device.hpu)


class IntTensor(CudaModuleRegister):
    """
    .. py:gpumgrcall:: hpu_modified

    Maps torch.cuda.IntTensor to torch.IntTensor + torch.Tensor.to("hpu"). See torch.cuda.FloatTensor for details.

    """

    is_replace_class = True
    tensor_type = torch.int32

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __new__(self, *args, **kwargs):
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
        G_LOGGER.info(
            api_type="hpu_modified",
            func_prefix="torch.cuda",
            old_args=log_args,
            new_call="torch.IntTensor(args={}, kwargs={{{}}}).to({})".format(
                args, _utils.kwargs_to_str(kwargs), Device.hpu
            ),
        )

        return torch.IntTensor(*args, **kwargs).to(Device.hpu)


class DoubleTensor(CudaModuleRegister):
    """
    .. py:gpumgrcall:: hpu_modified

    Maps torch.cuda.DoubleTensor to torch.DoubleTensor + torch.Tensor.to("hpu"). See torch.cuda.FloatTensor for details.

    """

    is_replace_class = True
    tensor_type = torch.float64

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __new__(self, *args, **kwargs):
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
        G_LOGGER.info(
            api_type="hpu_modified",
            func_prefix="torch.cuda",
            old_args=log_args,
            new_call="torch.IntTensor(args={}, kwargs={{{}}}).to({})".format(
                args, _utils.kwargs_to_str(kwargs), Device.hpu
            ),
        )

        return torch.FloatTensor(*args, **kwargs).to(Device.hpu)


class HalfTensor(CudaModuleRegister):
    """
    .. py:gpumgrcall:: hpu_modified

    Maps torch.cuda.HalfTensor to torch.HalfTensor + torch.Tensor.to("hpu"). See torch.cuda.FloatTensor for details.

    """

    is_replace_class = True
    tensor_type = torch.float16

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __new__(self, *args, **kwargs):
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
        G_LOGGER.info(
            api_type="hpu_modified",
            func_prefix="torch.cuda",
            old_args=log_args,
            new_call="torch.HalfTensor(args={}, kwargs={{{}}}).to({})".format(
                args, _utils.kwargs_to_str(kwargs), Device.hpu
            ),
        )

        return torch.HalfTensor(*args, **kwargs).to(Device.hpu)


class BFloat16Tensor(CudaModuleRegister):
    """
    .. py:gpumgrcall:: hpu_modified

    Maps torch.cuda.BFloat16Tensor to torch.BFloat16Tensor + torch.Tensor.to(“hpu”). See torch.cuda.FloatTensor for details.

    """

    is_replace_class = True
    tensor_type = torch.bfloat16

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __new__(self, *args, **kwargs):
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
        G_LOGGER.info(
            api_type="hpu_modified",
            func_prefix="torch.cuda",
            old_args=log_args,
            new_call="torch.BFloat16Tensor(args={}, kwargs={{{}}}).to({})".format(
                args, _utils.kwargs_to_str(kwargs), Device.hpu
            ),
        )

        return torch.BFloat16Tensor(*args, **kwargs).to(Device.hpu)


class ByteTensor(CudaModuleRegister):
    """
    .. py:gpumgrcall:: hpu_modified

    Maps torch.cuda.ByteTensor to torch.ByteTensor + torch.Tensor.to(“hpu”). See torch.cuda.FloatTensor for details.

    """

    is_replace_class = True
    tensor_type = torch.uint8

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __new__(self, *args, **kwargs):
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
        G_LOGGER.info(
            api_type="hpu_modified",
            func_prefix="torch.cuda",
            old_args=log_args,
            new_call="torch.ByteTensor(args={}, kwargs={{{}}}).to({})".format(
                args, _utils.kwargs_to_str(kwargs), Device.hpu
            ),
        )
        return torch.ByteTensor(*args, **kwargs).to(Device.hpu)


class CharTensor(CudaModuleRegister):
    """
    .. py:gpumgrcall:: hpu_modified

    Maps torch.cuda.CharTensor to torch.CharTensor + torch.Tensor.to(“hpu”). See torch.cuda.FloatTensor for details.

    """

    is_replace_class = True
    tensor_type = torch.int8

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __new__(self, *args, **kwargs):
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
        G_LOGGER.info(
            api_type="hpu_modified",
            func_prefix="torch.cuda",
            old_args=log_args,
            new_call="torch.CharTensor(args={}, kwargs={{{}}}).to({})".format(
                args, _utils.kwargs_to_str(kwargs), Device.hpu
            ),
        )
        return torch.CharTensor(*args, **kwargs).to(Device.hpu)


class ShortTensor(CudaModuleRegister):
    """
    .. py:gpumgrcall:: hpu_modified

    Maps torch.cuda.ShortTensor to torch.ShortTensor + torch.Tensor.to("hpu"). See torch.cuda.FloatTensor for details.

    """

    is_replace_class = True
    tensor_type = torch.int16

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __new__(self, *args, **kwargs):
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
        G_LOGGER.info(
            api_type="hpu_modified",
            func_prefix="torch.cuda",
            old_args=log_args,
            new_call="torch.ShortTensor(args={}, kwargs={{{}}}).to({})".format(
                args, _utils.kwargs_to_str(kwargs), Device.hpu
            ),
        )
        return torch.ShortTensor(*args, **kwargs).to(Device.hpu)


class LongTensor(CudaModuleRegister):
    """
    .. py:gpumgrcall:: hpu_modified

    Maps torch.cuda.LongTensor to torch.LongTensor + torch.Tensor.to("hpu"). See torch.cuda.FloatTensor for details.

    """

    is_replace_class = True
    tensor_type = torch.int64

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __new__(self, *args, **kwargs):
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
        G_LOGGER.info(
            api_type="hpu_modified",
            func_prefix="torch.cuda",
            old_args=log_args,
            new_call="torch.LongTensor(args={}, kwargs={{{}}}).to({})".format(
                args, _utils.kwargs_to_str(kwargs), Device.hpu
            ),
        )
        return torch.LongTensor(*args, **kwargs).to(Device.hpu)


class BoolTensor(CudaModuleRegister):
    """
    .. py:gpumgrcall:: hpu_modified

    Maps torch.cuda.BoolTensor to torch.BoolTensor + torch.Tensor.to(“hpu”). See torch.cuda.FloatTensor for details.

    """

    is_replace_class = True
    tensor_type = torch.bool

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __new__(self, *args, **kwargs):
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
        G_LOGGER.info(
            api_type="hpu_modified",
            func_prefix="torch.cuda",
            old_args=log_args,
            new_call="torch.BoolTensor(args={}, kwargs={{{}}}).to({})".format(
                args, _utils.kwargs_to_str(kwargs), Device.hpu
            ),
        )
        return torch.BoolTensor(*args, **kwargs).to(Device.hpu)
