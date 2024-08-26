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

from habana_frameworks.torch.gpu_migration.core.module_helper import \
    function_helper

try:
    import habana_frameworks.torch.hpex
except Exception:
    raise ImportError("Habana hpex package is not installed")


def _activate():
    from . import amp, normalization, optimizers, parallel
    from habana_frameworks.torch.gpu_migration.core.register import \
        BaseModuleRegister


    replace_functions = BaseModuleRegister.get_child_all_functions()
    return replace_functions


replace_functions = _activate()
function_helper.initialize(replace_functions)
