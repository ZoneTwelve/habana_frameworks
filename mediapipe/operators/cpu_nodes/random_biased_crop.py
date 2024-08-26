from habana_frameworks.mediapipe.backend.nodes import opnode_tensor_info
from habana_frameworks.mediapipe.operators.media_nodes import MediaCPUNode
from habana_frameworks.mediapipe.backend.utils import get_str_dtype
from habana_frameworks.mediapipe.media_types import dtype as dt
from habana_frameworks.mediapipe.media_types import mediaDeviceType as mdt
from habana_frameworks.mediapipe.backend.utils import array_from_ptr
import media_random_biased_crop as mrbc
import numpy as np
import time


class random_biased_crop(MediaCPUNode):
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
        if(fw_params.device != mdt.LEGACY):
            return

        self.patch_size = params['patch_size'].copy()
        self.patch_size_np = self.patch_size[::-1]
        self.over_sampling = params['over_sampling']
        self.num_channels = params['num_channels']
        self.seed = params['seed']
        self.num_workers = params['num_workers']
        self.cache_bboxes = params['cache_bboxes']

        if(self.num_workers < 1):
            raise ValueError("minimun one worker needed")
        if(self.num_workers > 8):
            raise ValueError("Num workers capped to 8")
        if len(self.patch_size) != 3:
            raise ValueError("3D patch size expected")
        self.batch_size = fw_params.batch_size
        self.queue_depth = fw_params.queue_depth
        self.rbc = mrbc.RandBalancedCrop(self.patch_size_np,
                                         self.batch_size,
                                         self.over_sampling,
                                         self.seed,
                                         self.num_workers,
                                         self.queue_depth,
                                         self.num_channels,
                                         self.cache_bboxes)

    def gen_output_info(self):
        """
        Method to generate output type information.

        :returns : output tensor information of type "opnode_tensor_info".
        """
        self.img_shape = self.patch_size.copy()
        self.img_shape.append(self.num_channels)
        self.img_shape.append(self.batch_size)
        self.img_shape_np = self.img_shape[::-1]
        self.img_dtype = dt.FLOAT32
        self.lbl_shape = self.patch_size.copy()
        self.lbl_shape.append(1)  # labels channels is taken as one
        self.lbl_shape.append(self.batch_size)
        self.lbl_shape_np = self.lbl_shape[::-1]
        self.lbl_dtype = dt.UINT8
        self.coord_shape = [6, self.batch_size]
        self.coord_shape_np = self.coord_shape[::-1]
        self.coord_dtype = dt.UINT32
        out_info = []
        o = opnode_tensor_info(self.img_dtype, np.array(
            self.img_shape, dtype=np.uint32), "")
        out_info.append(o)
        o = opnode_tensor_info(self.lbl_dtype, np.array(
            self.lbl_shape, dtype=np.uint32), "")
        out_info.append(o)
        o = opnode_tensor_info(self.coord_dtype, np.array(
            self.coord_shape, dtype=np.uint32), "")
        out_info.append(o)
        return out_info

    def __call__(self, img, lbl):
        """
        Callable class method.

        :params img: image data
        :params lbl: label data
        """
        self.rbc.call(img, lbl)
        outputs = self.rbc.call_get()

        img_sliced = array_from_ptr(outputs[0],
                                    "f4",
                                    tuple(self.img_shape_np))
        lbl_sliced = array_from_ptr(outputs[1],
                                    "u1",
                                    tuple(self.lbl_shape_np))
        coord = array_from_ptr(outputs[2],
                               "u4",
                               tuple(self.coord_shape_np))

        return img_sliced, lbl_sliced, coord
