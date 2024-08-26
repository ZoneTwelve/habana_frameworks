import torch
from torch import Tensor
from habana_frameworks.torch.gpu_migration.torch.nn.functional import FunctionalModuleRegister 
from habana_frameworks.torch.hpex.kernels import FusedSDPA
from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER
import habana_frameworks.torch.hpu as ht

@FunctionalModuleRegister.register_f("scaled_dot_product_attention")
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> Tensor:
    """
    .. py:gpumgrcall:: scaled_dot_product_attention.hpu_modified

    Uses Intel Gaudi FusedSDPA when any of math_sdp_enabled, mem_efficient_sdp_enabled and flash_sdp_enabled are True and native torch.nn.scaled_dot_product_attention otherwise. More information available under: https://docs.habana.ai/en/latest/PyTorch/Python_Packages.html#hpex-kernels-fusedsdpa

    """
    # based on: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    # torch.scaled_dot_product_attention has following 3 modes:
    # math_sdp_enabled - C++ implementation of attention - mappable directly to Habana FusedSDPA.
    # mem_efficient_sdp_enabled - optimizations from https://github.com/facebookresearch/xformers - mappable to recompute variant of Habana FusedSDPA.
    # flash_sdp_enabled - FlashAttention2 - same as above, either with or without recompute.
    # if any of the above is True the code will call FusedSDPA with recompute controlled through mem_efficient_sdp_enabled flag.

    math_sdp_enabled = torch.backends.cuda.math_sdp_enabled()
    mem_efficient_sdp_enabled = torch.backends.cuda.mem_efficient_sdp_enabled()
    flash_sdp_enabled = torch.backends.cuda.flash_sdp_enabled()

    G_LOGGER.info(
        api_type="hpu_modified",
        func_prefix="torch.nn.functional",
        new_call="FusedSDPA.apply",
    )

    if math_sdp_enabled == True or mem_efficient_sdp_enabled == True or flash_sdp_enabled == True:
        with ht.sdp_kernel(enable_recompute = mem_efficient_sdp_enabled):
            return FusedSDPA.apply(query, key, value, attn_mask, dropout_p, is_causal, scale)
    else:
        return FunctionalModuleRegister.call_parent_func("scaled_dot_product_attention", query, key, value, attn_mask, dropout_p, is_causal)

