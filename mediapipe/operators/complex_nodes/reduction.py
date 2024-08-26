from habana_frameworks.mediapipe.backend.nodes import opnode_tensor_info
from habana_frameworks.mediapipe.operators.media_nodes import MediaComplexNode
from habana_frameworks.mediapipe.operators.media_nodes import media_layout
from habana_frameworks.mediapipe.media_types import dtype as dt
from habana_frameworks.mediapipe.media_types import ftype as ft
from habana_frameworks.mediapipe.media_types import decoderType as dect
from habana_frameworks.mediapipe import fn  # NOQA
import numpy as np
import math as pmath


class reduce_min(MediaComplexNode):
    """
    Class defining media decoder node.

    """

    def __init__(self, name, device, params, node_attr, fw_params):
        """
        Constructor method.

        :params params: node specific params.
        :params node_attr: node output information
        """
        if (device == "cpu"):
            raise ValueError("CPU reduction op not supported")
        super().__init__(name, device, node_attr)
        if (len(node_attr) == 1):
            dtype = node_attr[0]["outputType"]
            output_zp = node_attr[0]["outputZp"]
            output_scale = node_attr[0]["outputScale"]
        else:
            dtype = []
            output_zp = []
            output_scale = []
            for i in range(len(node_attr)):
                dtype.append(node_attr[i]["outputType"])
                output_zp.append(node_attr[i]["outputZp"])
                output_scale.append(node_attr[i]["outputScale"])

        reductionDimension = params['reductionDimension']
        self.minOps = []

        if isinstance(reductionDimension, int):
            reductionDimension = [reductionDimension]

        for r in reductionDimension:
            self.minOps.append(fn._ReduceMin_(reductionDimension=r,
                                              dtype=dtype,
                                              output_zerop=output_zp,
                                              output_scale=output_scale,
                                              device=device))

    def __call__(self, images):
        """
        Callable class method.

        """
        img = images
        for m in self.minOps:
            img, idx = m(img)
        return img, idx


class reduce_max(MediaComplexNode):
    """
    Class defining media decoder node.

    """

    def __init__(self, name, device, params, node_attr, fw_params):
        """
        Constructor method.

        :params params: node specific params.
        :params node_attr: node output information
        """
        if (device == "cpu"):
            raise ValueError("CPU reduction op not supported")
        super().__init__(name, device, node_attr)
        if (len(node_attr) == 1):
            dtype = node_attr[0]["outputType"]
            output_zp = node_attr[0]["outputZp"]
            output_scale = node_attr[0]["outputScale"]
        else:
            dtype = []
            output_zp = []
            output_scale = []
            for i in range(len(node_attr)):
                dtype.append(node_attr[i]["outputType"])
                output_zp.append(node_attr[i]["outputZp"])
                output_scale.append(node_attr[i]["outputScale"])
        reductionDimension = params['reductionDimension']
        self.maxOps = []

        if isinstance(reductionDimension, int):
            reductionDimension = [reductionDimension]

        for r in reductionDimension:
            self.maxOps.append(fn._ReduceMax_(reductionDimension=r,
                                              dtype=dtype,
                                              output_zerop=output_zp,
                                              output_scale=output_scale,
                                              device=device))

    def __call__(self, images):
        """
        Callable class method.

        """
        img = images
        for m in self.maxOps:
            img, idx = m(img)
        return img, idx
