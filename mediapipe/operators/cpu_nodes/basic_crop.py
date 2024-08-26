from habana_frameworks.mediapipe.backend.nodes import opnode_tensor_info
from habana_frameworks.mediapipe.operators.media_nodes import MediaCPUNode
from habana_frameworks.mediapipe.backend.utils import get_str_dtype
from habana_frameworks.mediapipe.media_types import dtype as dt
import media_random_biased_crop as mrbc
import numpy as np
import time


class basic_crop(MediaCPUNode):
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
            name, None, device, inputs, params, cparams, node_attr, fw_params)
        self.patch_size = params['patch_size'].copy()
        self.patch_size_np = self.patch_size[::-1]
        self.num_channels = params['num_channels']
        self.center_crop = params['center_crop']
        self.out_dtype = node_attr[0]['outputType']
        if len(self.patch_size) != 3:
            raise ValueError("3D patch size expected")

        self.batch_size = fw_params.batch_size

    def gen_output_info(self):
        """
        Method to generate output type information.

        :returns : output tensor information of type "opnode_tensor_info".
        """
        self.out_shape = self.patch_size.copy()
        self.out_shape.append(self.num_channels)
        self.out_shape.append(self.batch_size)
        self.out_shape_np = self.out_shape[::-1]
        out_info = []
        o = opnode_tensor_info(self.out_dtype, np.array(
            self.out_shape, dtype=np.uint32), "")
        out_info.append(o)
        return out_info

    def __call__(self, inp):
        """
        Callable class method.

        :params img: image data
        :params lbl: label data
        """

        out_sliced = np.empty(shape=self.out_shape_np, dtype=self.out_dtype)

        for i in range(self.batch_size):
            """
            img_tmp = img[i][:,
                             0:128,
                             0:128,
                             0:128]
            lbl_tmp = lbl[i][:,
                             0:128,
                             0:128,
                             0:128]
            coord[i] = [0,128,
                        0,128,
                        0,128]
            """

            # print("img {} lbl {} coord ( {} ) ".format(
            #    img[i].shape, lbl[i].shape, coord[i]))

            if self.center_crop == True:
                img_shape_np = inp[i].shape  # (C,D,H,W)

                if len(img_shape_np) == 4:
                    img_shape_ar = np.array([img_shape_np[3], img_shape_np[2], img_shape_np[1]],
                                            dtype=np.int32)
                else:
                    raise ValueError("4 dim input expected")

                patch_size_ar = np.array(self.patch_size, dtype=np.int32)
                offset = ((img_shape_ar - patch_size_ar) + 1) // 2

                out_sliced[i] = inp[i][:,
                                       offset[2]:offset[2]+self.patch_size[2],
                                       offset[1]:offset[1]+self.patch_size[1],
                                       offset[0]:offset[0]+self.patch_size[0]]
            else:
                out_sliced[i] = inp[i][:,
                                       0:self.patch_size[2],
                                       0:self.patch_size[1],
                                       0:self.patch_size[0]]

        # end_time0 = time.perf_counter()
        # print("<<< Random biased crop {:.6f}".format(end_time0-start_time0))
        return out_sliced
