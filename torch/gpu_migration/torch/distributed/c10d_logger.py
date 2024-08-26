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

from typing import Any, Dict

import torch.distributed as dist
from . import DistributedLoggerModuleRegister
from habana_frameworks.torch.gpu_migration.torch import TORCH_VERSION

@DistributedLoggerModuleRegister.register_f("_get_msg_dict")
def _get_msg_dict(func_name, *args, **kwargs) -> Dict[str, Any]:
    """
    .. py:gpumgrcall:: _get_msg_dict.hpu_match

    Removes nccl_version from msg_dict.

    """
    if dist.is_initialized():
        msg_dict = {
            "func_name": f"{func_name}",
            "args": f"{args}, {kwargs}",
            "pg_name": f"{dist._get_process_group_name(kwargs.get('pg'))}",  # type: ignore[arg-type]
            "backend": f"{dist.get_backend(kwargs.get('group'))}",
            "world_size": f"{dist.get_world_size()}",
            "group_size": f"{dist.get_world_size(kwargs.get('group'))}",
            "global_rank": f"{dist.get_rank()}",
            "local_rank": f"{dist.get_rank(kwargs.get('group'))}",
        }
    else:
        msg_dict = {
            "func_name": f"{func_name}",
            "args": f"{args}, {kwargs}",
        }
    return msg_dict