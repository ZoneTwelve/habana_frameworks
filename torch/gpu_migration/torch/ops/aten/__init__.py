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
import torch
from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER
from habana_frameworks.torch.hpex.kernels import FusedSDPA
import habana_frameworks.torch.hpu as ht
from habana_frameworks.torch.utils.internal import is_lazy

if is_lazy():
    __all__ = ["_scaled_dot_product_flash_attention", "_scaled_dot_product_efficient_attention", "_scaled_dot_product_attention_math"]

from habana_frameworks.torch.gpu_migration.core.register import \
    BaseModuleRegister


class TorchOpsAtenRegister(BaseModuleRegister):
    @classmethod
    def get_module(cls):
        return torch.ops.aten

#torch.ops.aten._scaled_dot_product_flash_attention(query, key, value, dropout_p=dropout_p, is_causal=is_causal, scale=scale, return_debug_mask=True)
@TorchOpsAtenRegister.register_f("_scaled_dot_product_flash_attention")
def _scaled_dot_product_flash_attention(*args, **kwargs):
    """
    .. py:gpumgrcall:: _scaled_dot_product_flash_attention.hpu_modified

    Uses Habana FusedSDPA with enable_recompute = False. Ignores return_debug_mask argument.

    """
    kwargs.pop("return_debug_mask", None)

    G_LOGGER.info(
        api_type="hpu_modified",
        func_prefix="torch.ops.aten",
        new_call="FusedSDPA.apply",
    )

    with ht.sdp_kernel(enable_recompute = False):
        return FusedSDPA.apply(*args, **kwargs)


#torch.ops.aten._scaled_dot_product_efficient_attention(query, key, value, dropout_p=dropout_p, is_causal=is_causal, scale=scale, return_debug_mask=True)
@TorchOpsAtenRegister.register_f("_scaled_dot_product_efficient_attention")
def _scaled_dot_product_efficient_attention(*args, **kwargs):
    """
    .. py:gpumgrcall:: _scaled_dot_product_efficient_attention.hpu_modified

    Uses Habana FusedSDPA with enable_recompute = True. Ignores return_debug_mask argument.

    """
    kwargs.pop("return_debug_mask", None)

    G_LOGGER.info(
        api_type="hpu_modified",
        func_prefix="torch.ops.aten",
        new_call="FusedSDPA.apply",
    )

    with ht.sdp_kernel(enable_recompute = True):
        return FusedSDPA.apply(*args, **kwargs)

#torch.ops.aten._scaled_dot_product_attention_math(query_ref_lp, key_ref_lp, value_ref_lp, dropout_p=dropout_p, is_causal=is_causal, scale=scale, dropout_mask=dropout_mask)
@TorchOpsAtenRegister.register_f("_scaled_dot_product_attention_math")
def _scaled_dot_product_attention_math(*args, **kwargs):
    """
    .. py:gpumgrcall:: _scaled_dot_product_attention_math.hpu_modified

    Uses Habana FusedSDPA with enable_recompute = False. Ignores return_debug_mask argument.

    """
    kwargs.pop("return_debug_mask", None)

    G_LOGGER.info(
        api_type="hpu_modified",
        func_prefix="torch.ops.aten",
        new_call="FusedSDPA.apply",
    )

    with ht.sdp_kernel(enable_recompute = False):
        return FusedSDPA.apply(*args, **kwargs)