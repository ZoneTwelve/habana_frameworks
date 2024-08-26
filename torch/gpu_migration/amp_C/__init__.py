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

"""
The module to be replaced:
    - amp_C
"""

from typing import List, Optional, Tuple

import torch


def multi_tensor_adagrad(
    arg0: int,
    arg1: torch.Tensor,
    arg2: List[List[torch.Tensor]],
    arg3: float,
    arg4: float,
    arg5: int,
    arg6: float,
) -> None:
    """
    .. py:gpumgrcall:: multi_tensor_adagrad.hpu_mismatch

    Inactive Call.

    Description:
        We do not support float16 for now.
    """
    pass


def multi_tensor_adam(
    arg0: int,
    arg1: torch.Tensor,
    arg2: List[List[torch.Tensor]],
    arg3: float,
    arg4: float,
    arg5: float,
    arg6: float,
    arg7: int,
    arg8: int,
    arg9: int,
    arg10: float,
) -> None:
    """
    .. py:gpumgrcall:: multi_tensor_adam.hpu_mismatch

    Inactive Call.

    Description:
        We do not support float16 for now.
    """
    pass


def multi_tensor_axpby(
    arg0: int,
    arg1: torch.Tensor,
    arg2: List[List[torch.Tensor]],
    arg3: float,
    arg4: float,
    arg5: int,
) -> None:
    """
    .. py:gpumgrcall:: multi_tensor_axpby.hpu_mismatch

    Inactive Call.

    Description:
        We do not support float16 for now.
    """
    pass


def multi_tensor_l2norm(
    arg0: int, arg1: torch.Tensor, arg2: List[List[torch.Tensor]], arg3: Optional[bool]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    .. py:gpumgrcall:: multi_tensor_l2norm.hpu_mismatch

    Inactive Call.

    Description:
        We do not support float16 for now.
    """
    pass


def multi_tensor_l2norm_mp(
    arg0: int, arg1: torch.Tensor, arg2: List[List[torch.Tensor]], arg3: Optional[bool]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    .. py:gpumgrcall:: multi_tensor_l2norm_mp.hpu_mismatch

    Inactive Call.

    Description:
        We do not support float16 for now.
    """
    pass


def multi_tensor_l2norm_scale(
    arg0: int,
    arg1: torch.Tensor,
    arg2: List[List[torch.Tensor]],
    arg3: float,
    arg4: Optional[bool],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    .. py:gpumgrcall:: multi_tensor_l2norm_scale.hpu_mismatch

    Inactive Call.

    Description:
        We do not support float16 for now.
    """
    pass


def multi_tensor_lamb(
    arg0: int,
    arg1: torch.Tensor,
    arg2: List[List[torch.Tensor]],
    arg3: float,
    arg4: float,
    arg5: float,
    arg6: float,
    arg7: int,
    arg8: int,
    arg9: float,
    arg10: int,
    arg11: int,
    arg12: torch.Tensor,
    arg13: float,
    arg14: Optional[bool],
) -> None:
    """
    .. py:gpumgrcall:: multi_tensor_lamb.hpu_mismatch

    Inactive Call.

    Description:
        We do not support float16 for now.
    """
    pass


def multi_tensor_lamb_mp(
    arg0: int,
    arg1: torch.Tensor,
    arg2: List[List[torch.Tensor]],
    arg3: torch.Tensor,
    arg4: float,
    arg5: float,
    arg6: float,
    arg7: torch.Tensor,
    arg8: int,
    arg9: float,
    arg10: int,
    arg11: int,
    arg12: torch.Tensor,
    arg13: torch.Tensor,
    arg14: Optional[bool],
    arg15: torch.Tensor,
    arg16: torch.Tensor,
) -> None:
    """
    .. py:gpumgrcall:: multi_tensor_lamb_mp.hpu_mismatch

    Inactive Call.

    Description:
        We do not support float16 for now.
    """
    pass


def multi_tensor_lamb_stage1_cuda(
    arg0: int,
    arg1: torch.Tensor,
    arg2: List[List[torch.Tensor]],
    arg3: torch.Tensor,
    arg4: int,
    arg5: float,
    arg6: float,
    arg7: float,
    arg8: torch.Tensor,
    arg9: float,
) -> None:
    """
    .. py:gpumgrcall:: multi_tensor_lamb_stage1_cuda.hpu_mismatch

    Inactive Call.

    Description:
        We do not support float16 for now.
    """
    pass


def multi_tensor_lamb_stage2_cuda(
    arg0: int,
    arg1: torch.Tensor,
    arg2: List[List[torch.Tensor]],
    arg3: torch.Tensor,
    arg4: torch.Tensor,
    arg5: float,
    arg6: float,
    arg7: Optional[bool],
) -> None:
    """
    .. py:gpumgrcall:: multi_tensor_lamb_stage2_cuda.hpu_mismatch

    Inactive Call.

    Description:
        We do not support float16 for now.
    """
    pass


def multi_tensor_novograd(
    arg0: int,
    arg1: torch.Tensor,
    arg2: List[List[torch.Tensor]],
    arg3: torch.Tensor,
    arg4: float,
    arg5: float,
    arg6: float,
    arg7: float,
    arg8: int,
    arg9: int,
    arg10: float,
    arg11: int,
    arg12: int,
    arg13: int,
) -> None:
    """
    .. py:gpumgrcall:: multi_tensor_novograd.hpu_mismatch

    Inactive Call.

    Description:
        We do not support float16 for now.
    """
    pass


def multi_tensor_scale(
    arg0: int, arg1: torch.Tensor, arg2: List[List[torch.Tensor]], arg3: float
) -> None:
    """
    .. py:gpumgrcall:: multi_tensor_scale.hpu_mismatch

    Inactive Call.

    Description:
        We do not support float16 for now.
    """
    pass


def multi_tensor_sgd(
    arg0: int,
    arg1: torch.Tensor,
    arg2: List[List[torch.Tensor]],
    arg3: float,
    arg4: float,
    arg5: float,
    arg6: float,
    arg7: bool,
    arg8: bool,
    arg9: bool,
    arg10: float,
) -> None:
    """
    .. py:gpumgrcall:: multi_tensor_sgd.hpu_mismatch

    Inactive Call.

    Description:
        We do not support float16 for now.
    """
    pass
