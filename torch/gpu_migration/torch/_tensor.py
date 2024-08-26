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

import torch

import habana_frameworks.torch.gpu_migration.torch.cuda
from habana_frameworks.torch.gpu_migration.core import _utils
from habana_frameworks.torch.gpu_migration.core._enums import Device
from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER

from . import TorchModuleRegister


class Tensor(torch.Tensor, TorchModuleRegister):
    @classmethod
    def _save_orig_func_gpu_migration(cls):
        return ["to", "type", "numpy", "pin_memory"]

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def record_stream(self, s: torch.Stream):
        """
        .. py:gpumgrcall:: record_stream.hpu_mismatch

        Inactive Call.

        """
        # TODO: SW-115109
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
        G_LOGGER.info(
            api_type="hpu_mismatch",
            func_prefix="torch.Tensor",
            old_args=log_args,
            new_call="Dummy",
        )
        pass

    def to(self, *args, **kwargs):
        """
        .. py:gpumgrcall:: to.hpu_match

        Changes device arguments from “cuda” to “hpu” and dtype from torch.float16 to torch.bfloat16.

        """
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None

        new_args = list(args)
        new_kwargs = {key: val for key, val in kwargs.items()}

        device = kwargs.get("device")
        if device is None and len(args) >= 1:
            if (isinstance(args[0], str) or isinstance(args[0], torch.device)) and str(
                args[0]
            ).startswith(Device.cuda):
                to_device = str(args[0]).replace(Device.cuda, Device.hpu)
            elif isinstance(args[0], int):
                to_device = Device.hpu + ":" + str(args[0])
            else:
                to_device = args[0]
            new_args[0] = to_device
        elif device is not None:
            if str(device).startswith(Device.cuda):
                new_kwargs["device"] = str(device).replace(Device.cuda, Device.hpu)
            elif isinstance(device, int):
                new_kwargs["device"] = Device.hpu + ":" + str(device)

        if os.getenv("PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION") == "1":
            dtype = kwargs.get("dtype")
            if dtype is None and len(args) >= 1:
                if isinstance(args[0], torch.dtype) and dtype == torch.float16:
                    new_args[0] = torch.bfloat16
            elif dtype is not None:
                if dtype == torch.float16:
                    new_kwargs["dtype"] = torch.bfloat16
        else:
            pass

        new_args = tuple(new_args)

        if args != new_args or kwargs != new_kwargs:
            G_LOGGER.info(
                api_type="hpu_match",
                func_prefix="torch.Tensor",
                old_args=log_args,
                new_call="torch.Tensor.to(args={}, kwargs={{{}}})".format(
                    new_args, _utils.kwargs_to_str(new_kwargs)
                ),
            )

        return Tensor.call_parent_func("to", self, *new_args, **new_kwargs)

    def cuda(
        self, device=None, non_blocking=False, memory_format=torch.preserve_format
    ) -> torch.Tensor:
        """
        .. py:gpumgrcall:: cuda.hpu_match

        Changes the function to torch.Tensor.to("hpu").
        """
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None

        d = None
        if device is None:
            d = Device.hpu
        else:
            if str(device).startswith(Device.cuda):
                d = str(device).replace(Device.cuda, Device.hpu)
            elif str(device).isnumeric():
                d = Device.hpu + ":" + str(device)

        G_LOGGER.info(
            api_type="hpu_match",
            func_prefix="torch.Tensor",
            old_args=log_args,
            new_call=(
                "torch.Tensor.to(device={}, non_blocking={}, memory_format={})".format(
                    d, non_blocking, memory_format
                )
            ),
        )

        return self.to(device=d, non_blocking=non_blocking, memory_format=memory_format)

    def __getattribute__(self, name, /):
        if name == "is_cuda":
            return self.device.type == "hpu"
        return object.__getattribute__(self, name)

    def half(self, memory_format=torch.preserve_format):
        """
        .. py:gpumgrcall:: half.hpu_match

        Changes the function to torch.Tensor.to(torch.bfloat16).
        """
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None

        convert_fp16_to_bf16 = os.getenv("PT_HPU_CONVERT_FP16_TO_BF16_FOR_MIGRATION", "0") == "1"
        if convert_fp16_to_bf16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        G_LOGGER.info(
            api_type="hpu_match",
            func_prefix="torch.Tensor",
            old_args=log_args,
            new_call="torch.Tensor.to({}, memory_format={})".format(
                dtype, memory_format
            ),
        )

        return Tensor.call_parent_func("to", self, dtype, memory_format=memory_format)

    def numpy(self):
        """
        .. py:gpumgrcall:: numpy.hpu_modified

        If tensor datatype is torch.bfloat16, changes the function to torch.Tensor.to(torch.float16) and torch.Tensor.numpy.

        """
        tmp = self
        if self.dtype == torch.bfloat16:
            tmp = self.to(torch.float16)
        return Tensor.call_parent_func("numpy", tmp)

    def type(self, dtype=None, non_blocking=False, **kwargs):
        """
        .. py:gpumgrcall:: type.hpu_match

        Maps to torch.Tensor.type("torch.<Scalar>Tensor") and torch.Tensor.to("hpu") if dtype is torch.hpu.<Scalar>Tensor. If dtype is None this changes the tensor type to the same type as CUDA tensor.

        """
        #Changed Behavior:
        #    - If dtype is torch.hpu.<Scalar>Tensor, change this function to torch.Tensor.type("torch.<Scalar>Tensor") and torch.Tensor.to("hpu")
        #    - If dtype is None and self is a HPU tensor, change HPU tensor type to CUDA tensor type

        #Descriptions:
        #    >>> # The reason to change call is that user may check torch.Tensor.type() return value
        #    >>> supported_types = ["torch.cuda.HalfTensor",
        #            "torch.cuda.FloatTensor",
        #            "torch.cuda.DoubleTensor",
        #            "torch.cuda.BFloat16Tensor",
        #            SparseTensor.type()]
        #    >>> assert t.type() in supported_types
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None

        hpu_dtype_map = {
            "torch.hpu.FloatTensor": torch.float32,
            "torch.hpu.DoubleTensor": torch.float64,
            "torch.hpu.HalfTensor": torch.float16,
            "torch.hpu.BFloat16Tensor": torch.bfloat16,
            "torch.hpu.ByteTensor": torch.uint8,
            "torch.hpu.CharTensor": torch.int8,
            "torch.hpu.ShortTensor": torch.int16,
            "torch.hpu.IntTensor": torch.int32,
            "torch.hpu.LongTensor": torch.int64,
            "torch.hpu.BoolTensor": torch.bool,
        }
        # casts this object to the specified type
        if dtype is not None:

            if isinstance(dtype, str):
                if dtype.startswith("torch.cuda"):
                    # hpu tensor x and call x.type() will return torch.cuda.<Scalar>Tensor
                    dtype = dtype.replace("cuda", "hpu")
                if dtype.startswith("torch.hpu"):
                    G_LOGGER.info(
                        api_type="hpu_match",
                        func_prefix="torch.Tensor",
                        old_args=log_args,
                        new_call="torch.Tensor.to(device={}, dtype={})".format(
                            Device.hpu, hpu_dtype_map[dtype]
                        ),
                    )
                    # TODO: remove this case when SW-115106 is resolved
                    return self.to(device=Device.hpu, dtype=hpu_dtype_map[dtype])
            elif hasattr(dtype, "tensor_type"):
                G_LOGGER.info(
                    api_type="hpu_match",
                    func_prefix="torch.Tensor",
                    old_args=log_args,
                    new_call="torch.Tensor.to(device={}, dtype={})".format(
                        Device.hpu, dtype.tensor_type
                    ),
                )
                return self.to(device=Device.hpu, dtype=dtype.tensor_type)

            return Tensor.call_parent_func("type", self, dtype, non_blocking, **kwargs)

        # Returns the type
        if dtype is None and len(kwargs) == 0 and self.device.type == Device.hpu:
            ret = Tensor.call_parent_func("type", self, dtype, non_blocking, **kwargs)
            ret_cuda = ret.replace(Device.hpu, Device.cuda)

            G_LOGGER.info(
                api_type="hpu_match",
                func_prefix="torch.Tensor",
                old_args=log_args,
                new_call="change output value from {} to {}".format(ret, ret_cuda),
            )

            return ret_cuda

        return Tensor.call_parent_func("type", self, dtype, non_blocking, **kwargs)

    def pin_memory(self, device=None):
        """
        .. py:gpumgrcall:: pin_memory.hpu_match

        Pins memory to HPU device instead of GPU.

        """
        log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None

        d = device

        if device is None:
            d = Device.hpu
        elif device is not None:
            if str(device).startswith(Device.cuda):
                d = str(d).replace(Device.cuda, Device.hpu)
            elif isinstance(device, int):
                d = Device.hpu + ":" + str(device)

        if d != device:
            G_LOGGER.info(
                api_type="hpu_match",
                func_prefix="torch.Tensor",
                old_args=log_args,
                new_call="torch.Tensor.pin_memory(args={})".format(d),
            )

        return Tensor.call_parent_func("pin_memory", self, d)
