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
from habana_frameworks.torch.gpu_migration.torch.nn import NNModuleRegister
import torch.nn.functional as F

from typing import Optional


class Embedding(torch.nn.Embedding, NNModuleRegister):
    """
    .. py:gpumgrcall:: hpu_modified

    When using sparse gradients, the whole layer and inputs are moved to CPU.

    """

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.sparse is True:
            input = input.to("cpu")
            self = self.to("cpu")

            G_LOGGER.info(
                api_type="hpu_modified",
                func_prefix="torch.nn.Embedding",
                new_call="Currently not supporting sparse gradients. When using sparse gradients whole layer and inputs are moved to CPU.",
            )

        output = F.embedding(
                input, self.weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)


        if torch.hpu.is_available():
            output = output.to("hpu")
        return output


class EmbeddingBag(torch.nn.EmbeddingBag, NNModuleRegister):
    """
    .. py:gpumgrcall:: hpu_modified

    When using sparse gradients, the whole layer and inputs are moved to CPU.

    """

    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def forward(self, input: torch.Tensor, offsets: Optional[torch.Tensor] = None,
                per_sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.sparse is True:
            input = input.to("cpu")
            self = self.to("cpu")
            if offsets is not None:
                offsets = offsets.to("cpu")
            if per_sample_weights is not None:
                per_sample_weights = per_sample_weights.to("cpu")
            G_LOGGER.info(
                api_type="hpu_modified",
                func_prefix="torch.nn.Embedding",
                new_call="Currently not supporting sparse gradients. When using sparse gradients whole layer and inputs are moved to CPU.",
            )

        output = F.embedding_bag(
                input, self.weight, offsets,
                self.max_norm, self.norm_type,
                self.scale_grad_by_freq, self.mode, self.sparse,
                per_sample_weights, self.include_last_offset,
                self.padding_idx)


        if torch.hpu.is_available():
            output = output.to("hpu")
        return output