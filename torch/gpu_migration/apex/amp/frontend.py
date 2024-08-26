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
import apex
import torch

from . import APEXAMPModuleRegister


def wrap_fwd(model):
    org_fwd = model.forward

    def wrap(*args, **kwargs):
        with torch.autocast(device_type="hpu", dtype=None):
            results = org_fwd(*args, **kwargs)
        return results

    model.forward = wrap
    return model


@APEXAMPModuleRegister.register_f()
def initialize(
    models,
    optimizers=None,
    enabled=True,
    opt_level="O1",
    cast_model_type=None,
    patch_torch_functions=None,
    keep_batchnorm_fp32=None,
    master_weights=None,
    loss_scale=None,
    cast_model_outputs=None,
    num_losses=1,
    verbosity=1,
    min_loss_scale=None,
    max_loss_scale=16777216.0,
):
    """
    .. py:gpumgrcall:: initialize.hpu_modified

    Uses native torch.autocast with device_type="hpu" instead.

    """
    if opt_level == "O3" or opt_level == "O2":
        opt_level = "O1"

    apex.amp._amp_state.opt_properties = apex.amp.frontend.Properties()
    apex.amp._amp_state.opt_properties.enabled = False
    apex.amp._amp_state.loss_scalers = []
    for _ in range(num_losses):
        apex.amp._amp_state.loss_scalers.append(apex.amp.scaler.LossScaler(1.0))

    amp_models = []
    if opt_level != "O0":
        if isinstance(models, torch.nn.Module):
            return wrap_fwd(models), optimizers
        elif isinstance(models, list):
            for model in models:
                amp_models.append(wrap_fwd(model))
        else:
            raise TypeError("models must be either a single model or a list of models.")
    else:
        amp_models = models

    return amp_models, optimizers
