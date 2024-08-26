from habana_frameworks.mediapipe.backend.nodes import opnode_tensor_info
from habana_frameworks.mediapipe.operators.media_nodes import MediaCPUNode
from habana_frameworks.mediapipe.backend.utils import get_str_dtype
from habana_frameworks.mediapipe.media_types import dtype as dt
import media_random_biased_crop as mrbc
import numpy as np
import math as pmath
import time


class gaussian_filter(MediaCPUNode):
    """
    Class representing media random biased crop cpu node.

    """

    def __init__(self, name, guid, device, inputs, params, cparams, node_attr, fw_params):
        """
        Constructor method.

        :params name: node name.
        :params guid: guid of node.
        :params guid: device on which this node should execute.
        :params params: node specific params.
        :params cparams: backend params.
        :params node_attr: node output information
        """
        super().__init__(
            name, guid, device, inputs, params, cparams, node_attr, fw_params)
        if (fw_params.device != "legacy"):
            return
        self.channels = params['channels']
        self.min_sigma = params['min_sigma']
        self.max_sigma = params['max_sigma']
        self.depth = params['input_depth']
        self.kSize = int(2 * pmath.ceil(3 * self.max_sigma) + 1)
        self.batch_size = fw_params.batch_size

    def gen_output_info(self):
        """
        Method to generate output type information.

        :returns : output tensor information of type "opnode_tensor_info".s
        """
        out_info = []
        o = opnode_tensor_info(dt.FLOAT32, np.array(
            [(self.depth * self.channels * self.batch_size), 1, self.kSize, 1], dtype=np.uint32), "")
        out_info.append(o)
        return out_info

    def __call__(self, sigmas):
        """
        Callable class method.

        :params img: image data
        :params lbl: label data
        """
        a = self.create_gaussian_kernel(sigmas)
        return a

    def create_gaussian_kernel(self, sigmas):
        gaussianWeights = self.create_oneD_gaussian_kernel(sigmas)
        gaussianWeights_np = np.array(gaussianWeights, dtype=np.float32)
        gaussianWeights_np = np.transpose(gaussianWeights_np)
        gaussianWeights_np = np.tile(
            gaussianWeights_np, self.channels*self.depth)
        gaussianWeights_np = np.expand_dims(gaussianWeights_np, axis=0)
        gaussianWeights_np = np.expand_dims(gaussianWeights_np, axis=2)
        return gaussianWeights_np

    def create_oneD_gaussian_kernel(self, sigmas):
        # Compute 1D Gaussian Filter of shape based on current sigma
        maxSizeOneD = self.kSize
        gaussianWeights = []
        for sigma in sigmas:
            if (sigma == 0):  # Do not blur
                weightG = [0.0]*maxSizeOneD
                mid = maxSizeOneD//2
                weightG[mid] = 1.0
                gaussianWeights.append(weightG)
            else:
                sizeOneD = 2 * pmath.ceil(3 * sigma) + 1
                r = int((sizeOneD - 1) / 2)
                exp_scale = 0.5 / (sigma * sigma)
                sum = 0.0
                # Calculate first half
                weightG = [0.0]*sizeOneD

                for x in range(-r, 0):
                    weightG[x+r] = pmath.exp(-(x * x * exp_scale))
                    sum += weightG[x+r]

                # Total sum, it's symmetric with `1` in the center.
                sum *= 2.0
                sum += 1.0
                scale = 1.0 / float(sum)
                # place center, scaled element
                weightG[r] = scale
                # scale all elements so they sum up to 1, duplicate the second half
                for x in range(0, r):
                    weightG[x] *= scale
                    weightG[2 * r - x] = weightG[x]
                # make length equal to max length
                lenDiff = maxSizeOneD - len(weightG)
                if (lenDiff > 0):
                    halfLenDiff = int(lenDiff/2)
                    prefixZeros = [0]*halfLenDiff
                    suffixZeros = [0]*(lenDiff-halfLenDiff)
                    weightG = prefixZeros+weightG+suffixZeros
                gaussianWeights.append(weightG)
        return gaussianWeights
