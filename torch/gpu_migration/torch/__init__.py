###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import torch

from habana_frameworks.torch.gpu_migration.core.register import \
    BaseModuleRegister


class TorchModuleRegister(BaseModuleRegister):
    @classmethod
    def get_module(cls):
        return torch


# Check Torch Version
from packaging import version

TORCH_VERSION = version.Version(torch.__version__).base_version
supported_pt_versions = ['2.3.0', '2.3.1', '2.4.0']
assert TORCH_VERSION in supported_pt_versions

torch.version.cuda = "11.8"

# Disable JIT Since it is not supported on HPU
import os

disable_jit = os.getenv("DISABLE_JIT", "1")
if disable_jit == "1":
    torch.jit._state.disable()

import builtins
import os
from typing import Callable, Dict, Optional, Union
from habana_frameworks.torch.utils.internal import is_lazy



@TorchModuleRegister.register_f("compile")
def compile(model: Optional[Callable] = None, *,
            fullgraph: builtins.bool = False,
            dynamic: Union[builtins.bool, None] = None,
            backend: Union[str, Callable] = "inductor",
            mode: Union[str, None] = None,
            options: Optional[Dict[str, Union[str, builtins.int, builtins.bool]]] = None,
            disable: builtins.bool = False) -> Callable:
            """
            .. py:gpumgrcall:: compile.hpu_match

            Maps inductor backend to hpu_backend. For further details, refer to `:ref:`pytorch_known_issues``.

            """
            if backend == "inductor":
                backend = "hpu_backend"
            kwargs = {'fullgraph': fullgraph, 'dynamic': dynamic, 'backend': backend, 'mode': mode, 'options': options, 'disable': disable}
            torch.jit._state.enable()
            return TorchModuleRegister.call_parent_func("compile", model,  **kwargs)

import habana_frameworks.torch.gpu_migration.torch.cuda
import habana_frameworks.torch.gpu_migration.torch.distributed
import habana_frameworks.torch.gpu_migration.torch.nn
if is_lazy():
    import habana_frameworks.torch.gpu_migration.torch.ops.aten
import habana_frameworks.torch.gpu_migration.torch.optim
import habana_frameworks.torch.gpu_migration.torch.utils
from habana_frameworks.torch.gpu_migration.torch._C import *
from habana_frameworks.torch.gpu_migration.torch._tensor import Tensor
from habana_frameworks.torch.gpu_migration.torch.amp import *
from habana_frameworks.torch.gpu_migration.torch.serialization import load
