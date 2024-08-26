from habana_frameworks.mediapipe.backend.nodes import opnode_tensor_info
from habana_frameworks.mediapipe.operators.media_nodes import MediaComplexNode
from habana_frameworks.mediapipe.operators.media_nodes import media_layout
from habana_frameworks.mediapipe.media_types import dtype as dt
from habana_frameworks.mediapipe.media_types import ftype as ft
from habana_frameworks.mediapipe.media_types import decoderType as dect
from habana_frameworks.mediapipe.media_types import mediaDeviceType as mdt
from habana_frameworks.mediapipe import fn  # NOQA
import numpy as np
import math as pmath


def get_pad_params(W, H, kW, kH, S=1):
    # compute Pad values
    # (padT+padB) = (H * S - S - H + KH)
    padTB = H * S - S - H + kH
    # padL + padR = (W * S - S - W + KW)
    padLR = W * S - S - W + kW
    padL = padLR//2
    padR = padLR - padL
    padT = padTB//2
    padB = padTB - padT
    return padL, padR, padT, padB


class gaussian_blur(MediaComplexNode):
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
        # node_attr["outputType"]
        # node_attr["outputZp"]
        # node_attr["outputScale"]
        self.fw_device = fw_params.device
        max_sigma = params['max_sigma']
        min_sigma = params['min_sigma']
        shape = params['shape']  # [W,H,D,C,N]
        self.in_dtype = node_attr[0]["outputType"]
        if (len(shape) != 5):
            raise ValueError("Only 5 dimensional tensor supported")
        batch_size = shape[4]
        channels = shape[3]
        depth = shape[2]
        height = shape[1]
        width = shape[0]
        kW = int(2 * pmath.ceil(3 * max_sigma) + 1)

        # ---------------------first convolution-----------------------------
        self.gaussian_kernel1 = fn.GaussianFilter(name="gaussian_kernel1",
                                                  channels=channels,
                                                  min_sigma=min_sigma,
                                                  max_sigma=max_sigma,
                                                  input_depth=depth,
                                                  device="cpu")
        self.transp11 = fn.Transpose(name="transp11",permutation=[4, 3, 2, 0, 1],
                                     tensorDim=5,
                                     dtype=self.in_dtype,
                                     device="hpu")  # WHDCN -> NCDWH
        self.reshape11 = fn.Reshape(name="reshape11",size=[batch_size * channels * depth, width,
                                    height, 1],
                                    tensorDim=4,
                                    layout='',
                                    dtype=self.in_dtype,
                                    device="hpu")  # (NCD),W,H,1
        padL1, padR1, padT1, padB1 = get_pad_params(width, height, kW, 1)
        self.pad1 = fn.Pad(name="pad1",mode=1,
                           pads=[0, padL1, padT1, 0, 0, padR1, padB1, 0],
                           dtype=self.in_dtype,
                           device="hpu")
        if (fw_params.device != mdt.LEGACY):
            self.gk_transpose1 = fn.Transpose(permutation=[2, 1, 0],
                                              tensorDim=3,
                                              dtype=dt.FLOAT32,
                                              device='hpu')  # N,(C*D),kW

            self.gk_reshape10 = fn.Reshape(size=[channels*depth*batch_size, kW],
                                           tensorDim=2,
                                           layout='',
                                           dtype=dt.FLOAT32,
                                           device='hpu')  # kernel: [(NCD),kW]

            self.gk_reshape11 = fn.Reshape(size=[channels *
                                                 depth*batch_size, 1, kW, 1],
                                           tensorDim=4,
                                           layout='',
                                           dtype=dt.FLOAT32,
                                           device='hpu')  # kernel: [(NCD),1,kW,1]
        self.cast = fn.Cast(name="gaussian_kernel1",dtype=self.in_dtype, device='hpu')
        self.spatial_conv1 = fn.SpatialConv(name="spatial_conv1",nGroups=batch_size * channels * depth,
                                            kW=kW,
                                            dtype=self.in_dtype,
                                            device='hpu')

        self.reshape12 = fn.Reshape(name="reshape12",size=[batch_size, channels, depth, width, height],
                                    tensorDim=5,
                                    layout='',
                                    dtype=self.in_dtype,
                                    device='hpu')  # N,C,D,W,H
        self.transp12 = fn.Transpose(name="transp12",permutation=[3, 4, 2, 1, 0],
                                     tensorDim=5,
                                     dtype=self.in_dtype,
                                     device='hpu')  # NCDWH -> WHDCN
        # -----------------------second convolution--------------------------------------------------------
        self.gaussian_kernel2 = fn.GaussianFilter(name="gaussian_kernel2",channels=channels,
                                                  min_sigma=min_sigma,
                                                  max_sigma=max_sigma,
                                                  input_depth=depth,
                                                  device="cpu")
        self.transp21 = fn.Transpose(name="transp21",permutation=[4, 3, 2, 1, 0],
                                     tensorDim=5,
                                     dtype=self.in_dtype,
                                     device='hpu')  # WHDCN -> NCDHW
        self.reshape21 = fn.Reshape(name="reshape21",size=[batch_size * channels * depth, height,
                                    width, 1],
                                    tensorDim=4,
                                    layout='',
                                    dtype=self.in_dtype,
                                    device='hpu')  # (NCD)HW1
        padL2, padR2, padT2, padB2 = get_pad_params(height, width, kW, 1)
        self.pad2 = fn.Pad(name="pad2",mode=1,
                           pads=[0, padL2, padT2, 0, 0, padR2, padB2, 0],
                           dtype=self.in_dtype,
                           device='hpu')

        if (fw_params.device != mdt.LEGACY):
            self.gk_transpose2 = fn.Transpose(name="gaussian_kernel1",permutation=[2, 1, 0],
                                              tensorDim=3,
                                              dtype=dt.FLOAT32,
                                              device='hpu')  # N,(C*D),kW

            self.gk_reshape20 = fn.Reshape(size=[channels*depth*batch_size, kW],
                                           tensorDim=2,
                                           layout='',
                                           dtype=dt.FLOAT32,
                                           device='hpu')  # kernel: [(NCD),kW]

            self.gk_reshape21 = fn.Reshape(size=[channels *
                                                 depth*batch_size, 1, kW, 1],
                                           tensorDim=4,
                                           layout='',
                                           dtype=dt.FLOAT32,
                                           device='hpu')  # kernel: [(NCD),1,kW,1]

        self.spatial_conv2 = fn.SpatialConv(name="spatial_conv2",nGroups=batch_size * channels * depth,
                                            kW=kW,
                                            dtype=self.in_dtype,
                                            device='hpu')  # (NCD),H,W,1

        self.reshape22 = fn.Reshape(name="reshape22",size=[batch_size, channels, depth, height, width],
                                    tensorDim=5,
                                    layout='',
                                    dtype=self.in_dtype,
                                    device='hpu')  # N,C,D,H,W
        self.transp22 = fn.Transpose(name="transp22",permutation=[4, 3, 2, 1, 0],
                                     tensorDim=5,
                                     dtype=self.in_dtype,
                                     device='hpu')  # N,C,D,H,W -> WHDCN
        # -----------------------3rd convolution----------------------------------------------------------------------------
        self.gaussian_kernel3 = fn.GaussianFilter(name="gaussian_kernel3",channels=channels,
                                                  min_sigma=min_sigma,
                                                  max_sigma=max_sigma,
                                                  input_depth=height,
                                                  device="cpu")
        self.transp31 = fn.Transpose(name="transp31",permutation=[4, 3, 1, 2, 0],
                                     tensorDim=5,
                                     dtype=self.in_dtype,
                                     device='hpu')  # WHDCN -> NCHDW
        self.reshape31 = fn.Reshape(name="reshape31",size=[batch_size * channels * height, depth,
                                    width, 1],
                                    tensorDim=4,
                                    layout='',
                                    dtype=self.in_dtype,
                                    device='hpu')  # (NCH)DW1
        padL3, padR3, padT3, padB3 = get_pad_params(depth, width, kW, 1)
        self.pad3 = fn.Pad(name="pad3",mode=1,
                           pads=[0, padL3, padT3, 0, 0, padR3, padB3, 0],
                           dtype=self.in_dtype,
                           device='hpu')

        if (fw_params.device != mdt.LEGACY):
            self.gk_transpose3 = fn.Transpose(permutation=[2, 1, 0],
                                              tensorDim=3,
                                              dtype=dt.FLOAT32,
                                              device='hpu')  # N,(C*D),kW

            self.gk_reshape30 = fn.Reshape(size=[channels*height*batch_size, kW],
                                           tensorDim=2,
                                           layout='',
                                           dtype=dt.FLOAT32,
                                           device='hpu')  # kernel: [(NCD),kW]

            self.gk_reshape31 = fn.Reshape(size=[channels *
                                                 height*batch_size, 1, kW, 1],
                                           tensorDim=4,
                                           layout='',
                                           dtype=dt.FLOAT32,
                                           device='hpu')  # kernel: [(NCD),1,kW,1]

        self.spatial_conv3 = fn.SpatialConv(name="spatial_conv3",nGroups=batch_size * channels * height,
                                            kW=kW,
                                            dtype=self.in_dtype,
                                            device='hpu')  # (NCH)DW1

        self.reshape32 = fn.Reshape(name="reshape32",size=[batch_size, channels, height, depth, width],
                                    tensorDim=5,
                                    layout='',
                                    dtype=self.in_dtype,
                                    device='hpu')  # N,C,H,D,W
        self.transp32 = fn.Transpose(name="transp32",permutation=[4, 2, 3, 1, 0],
                                     tensorDim=5,
                                     dtype=self.in_dtype,
                                     device='hpu')  # N,C,H,D,W -> WHDCN

    def __call__(self, images, sigmas):
        """
        Callable class method.

        """
        # ------------conv1 along W --------------------------------
        images = self.transp11(images)  # WHDCN -> NCDWH
        images = self.reshape11(images)  # (NCD),W,H,1
        images = self.pad1(images)
        gaussian_k1 = self.gaussian_kernel1(sigmas)  # FLOAT32
        if (self.fw_device  != mdt.LEGACY):
            gaussian_k1 = self.gk_transpose1(gaussian_k1)  # FLOAT32
            gaussian_k1 = self.gk_reshape10(gaussian_k1)  # FLOAT32
            gaussian_k1 = self.gk_reshape11(gaussian_k1)  # FLOAT32
        if (self.in_dtype != dt.FLOAT32):
            gaussian_k1 = self.cast(gaussian_k1)
        images = self.spatial_conv1(images, gaussian_k1)  # (NCD),W,H,1
        images = self.reshape12(images)  # N,C,D,W,H
        images = self.transp12(images)  # NCDWH -> WHDCN

        # ------------conv2 along H --------------------------------
        images = self.transp21(images)  # WHDCN -> NCDHW
        images = self.reshape21(images)  # (NCD)HW1
        images = self.pad2(images)
        gaussian_k2 = self.gaussian_kernel2(sigmas)  # FLOAT32
        if (self.fw_device  != mdt.LEGACY):
            gaussian_k2 = self.gk_transpose2(gaussian_k2)  # FLOAT32
            gaussian_k2 = self.gk_reshape20(gaussian_k2)  # FLOAT32
            gaussian_k2 = self.gk_reshape21(gaussian_k2)  # FLOAT32
        if (self.in_dtype != dt.FLOAT32):
            gaussian_k2 = self.cast(gaussian_k2)
        images = self.spatial_conv2(images, gaussian_k2)  # (NCD),H,W,1
        images = self.reshape22(images)  # N,C,D,H,W
        images = self.transp22(images)  # N,C,D,H,W -> WHDCN
        # -----------conv3 along D--------------------------------
        images = self.transp31(images)  # WHDCN -> NCHDW
        images = self.reshape31(images)  # (NCH)DW1
        images = self.pad3(images)
        gaussian_k3 = self.gaussian_kernel3(sigmas)  # FLOAT32
        if (self.fw_device  != mdt.LEGACY):
            gaussian_k3 = self.gk_transpose3(gaussian_k3)  # FLOAT32
            gaussian_k3 = self.gk_reshape30(gaussian_k3)  # FLOAT32
            gaussian_k3 = self.gk_reshape31(gaussian_k3)  # FLOAT32
        if (self.in_dtype != dt.FLOAT32):
            gaussian_k3 = self.cast(gaussian_k3)
        images = self.spatial_conv3(images, gaussian_k3)  # (NCH)DW1
        images = self.reshape32(images)  # N,C,H,D,W
        images = self.transp32(images)  # N,C,H,D,W -> WHDCN
        return images
