from habana_frameworks.mediapipe.backend.nodes import opnode_tensor_info
from habana_frameworks.mediapipe.operators.media_nodes import MediaComplexNode
from habana_frameworks.mediapipe.operators.media_nodes import media_layout
from habana_frameworks.mediapipe.media_types import dtype as dt
from habana_frameworks.mediapipe.media_types import ftype as ft
from habana_frameworks.mediapipe.media_types import decoderType as dect
from habana_frameworks.mediapipe import fn  # NOQA
import numpy as np
import math as pmath


class random_flip(MediaComplexNode):
    """
    Class defining media decoder node.

    """

    def __init__(self, name, device, params, node_attr, fw_params):
        """
        Constructor method.

        :params params: node specific params.
        :params node_attr: node output information
        """

        super().__init__(name, device, node_attr)
        dtype = []
        output_zp = []
        output_scale = []
        if (len(node_attr) == 1):
            dtype.append(node_attr[0]["outputType"])
            output_zp.append(node_attr[0]["outputZp"])
            output_scale.append(node_attr[0]["outputScale"])
            dtype.append(dt.UINT8)
            output_zp.append(1)
            output_scale.append(0)
        else:
            dtype = []
            output_zp = []
            output_scale = []
            for i in range(len(node_attr)):
                dtype.append(node_attr[i]["outputType"])
                output_zp.append(node_attr[i]["outputZp"])
                output_scale.append(node_attr[i]["outputScale"])
        if device == "hpu":
            self.reshape = fn.Reshape(size=[fw_params.batch_size],
                                      tensorDim=1,
                                      dtype=dtype[1],
                                      output_zerop=output_zp[1],
                                      output_scale=output_scale[1],
                                      device="hpu")

        self.rflip = fn._random_flip_(**params,
                                      device=device,
                                      dtype=dtype[0],
                                      output_zerop=output_zp[0],
                                      output_scale=output_scale[0])

    def __call__(self, inp, predicate):
        """
        Callable class method.

        """
        if self.device == "hpu":
            predicate = self.reshape(predicate)

        out = self.rflip(inp, predicate)
        return out
