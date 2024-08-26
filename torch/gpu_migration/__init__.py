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
    import habana_frameworks.torch.hpu

    import habana_frameworks.torch
except Exception:
    raise ImportError("Habana python package is not installed")

import warnings

import pkg_resources


def _installed_packages_set():
    return {pkg.key for pkg in pkg_resources.working_set}


def _conditionally_import_apex():
    """
    This function imports gpu_migration.apex  if apex is installed in the system.
    This also imports the _C part of apex as a system wide modules.
    In order to avoid RuntimeError: apex.optimizers.FusedAdam requires cuda extensions
    apex.optimizers are also imported in the same way.
    """
    if "apex" in _installed_packages_set():
        from . import amp_C, apex, apex_C

        exist_libs = ["apex.optimizers", "apex_C", "amp_C"]
        for lib in exist_libs:
            function_helper.do_import(lib)
    else:
        warnings.warn(
            "apex not installed, gpu_migration will not swap api for this package."
        )


def _activate():
    from . import torch

    _conditionally_import_apex()
    from habana_frameworks.torch.gpu_migration.core.register import \
        BaseModuleRegister

    replace_functions = BaseModuleRegister.get_child_all_functions()
    return replace_functions


replace_functions = _activate()
function_helper.initialize(replace_functions)


def configure_logger():
    import os

    log_level = os.getenv("GPU_MIGRATION_LOG_LEVEL")

    if log_level is not None:
        from .core.logger import G_LOGGER

        if log_level == "3":
            G_LOGGER.module_severity = G_LOGGER.INFO
        elif log_level == "2":
            G_LOGGER.module_severity = G_LOGGER.VERBOSE
        elif log_level == "1":
            G_LOGGER.module_severity = G_LOGGER.EXTRA_VERBOSE

        G_LOGGER.log_file = "gpu_migration_{}.log".format(os.getpid())


configure_logger()
