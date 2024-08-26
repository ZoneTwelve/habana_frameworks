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

from . import CudaModuleRegister
from habana_frameworks.torch.gpu_migration.core.register import BaseModuleRegister
from habana_frameworks.torch.gpu_migration.core.logger import G_LOGGER
import torch


@CudaModuleRegister.register_f()
def make_graphed_callables(callables, sample_args, num_warmup_iters=3, allow_unused_input=False, pool=None):
    """
    .. py:gpumgrcall:: make_graphed_callables.hpu_match

    Maps torch.cuda.make_graphed_callables to habana_frameworks.torch.hpu.graphs.make_graphed_callables.
    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    G_LOGGER.info(api_type="hpu_match", func_prefix="torch.cuda", old_args=log_args,
                new_call="torch.hpu.make_graphed_callables({}, {}, num_warmup_iters={})".format(callables, sample_args, num_warmup_iters))

    from habana_frameworks.torch.hpu.graphs import make_graphed_callables
    return make_graphed_callables(callables, sample_args, num_warmup_iters, allow_unused_input)


@CudaModuleRegister.register_f()
def graph(cuda_graph, pool=None, stream=None, capture_error_mode: str = 'global'):
    """
    .. py:gpumgrcall:: graph.hpu_match

    Maps torch.cuda.graph to habana_frameworks.torch.hpu.graphs.graph.
    """
    log_args = locals() if G_LOGGER.module_severity <= G_LOGGER.INFO else None
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        old_args=log_args,
        new_call="torch.hpu.graph({}, pool={}, stream={})".format(
            cuda_graph, pool, stream
        ),
    )

    from habana_frameworks.torch.hpu.graphs import graph

    return graph(hpu_graph=cuda_graph, stream=stream)


@CudaModuleRegister.register_f()
def CUDAGraph():
    """
    .. py:gpumgrcall:: CUDAGraph.hpu_match

    Maps torch.cuda.CUDAGraph to habana_frameworks.torch.hpu.graphs.HPUGraph.
    """
    G_LOGGER.info(
        api_type="hpu_match", func_prefix="torch.cuda", new_call="torch.hpu.HPUGraph()"
    )

    from habana_frameworks.torch.hpu.graphs import HPUGraph

    return HPUGraph()


@CudaModuleRegister.register_f()
def graph_pool_handle():
    """
    .. py:gpumgrcall:: graph_pool_handle.hpu_mismatch

    Returns None.

    """
    G_LOGGER.info(api_type="hpu_mismatch", func_prefix="torch.cuda", new_call="Dummy")
    return None


@CudaModuleRegister.register_f()
def is_current_stream_capturing():
    """
    .. py:gpumgrcall:: is_current_stream_capturing.hpu_match

    Maps torch.cuda.graphs.is_current_stream_capturing to torch.hpu.graphs.is_current_stream_capturing.

    """
    G_LOGGER.info(
        api_type="hpu_match",
        func_prefix="torch.cuda",
        new_call="torch.hpu.is_current_stream_capturing()",
    )

    from habana_frameworks.torch.hpu.graphs import is_current_stream_capturing

    return is_current_stream_capturing()
