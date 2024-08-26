from habana_frameworks.mediapipe.backend.nodes import opnode_tensor_info
from habana_frameworks.mediapipe.operators.media_nodes import MediaCPUNode
from habana_frameworks.mediapipe.media_types import dtype as dt
from habana_frameworks.mediapipe.backend.utils import array_from_ptr
import media_random_flip_cpu as mrf
import numpy as np
import time


class random_flip_cpu(MediaCPUNode):
    """
    Class representing media random flip cpu node.

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
        self.horizontal = params['horizontal']
        self.vertical = params['vertical']
        self.depthwise = params['depthwise']
        self.num_channels = params['num_channels']
        self.num_workers = params['num_workers']
        if len(self.patch_size) != 3:
            raise ValueError("3D image shpe expected")
        if(self.num_workers < 1):
            raise ValueError("At least 1 worker is required")
        if(self.num_workers > 8):
            raise ValueError("Num workers capped to 8")
        if (self.horizontal == 0 and self.vertical == 0 and self.depthwise == 0):
            raise ValueError("At least one flip type should be mentioned")
        self.queue_depth = fw_params.queue_depth
        self.batch_size = fw_params.batch_size
        self.random_flip = mrf.RandomFlipCpu(self.patch_size_np,
                                             self.num_channels,
                                             self.batch_size,
                                             self.num_workers,
                                             self.queue_depth,
                                             self.horizontal,
                                             self.vertical,
                                             self.depthwise)

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
        out_info = []
        o = opnode_tensor_info(self.img_dtype, np.array(
            self.img_shape, dtype=np.uint32), "")
        out_info.append(o)
        return out_info

    def __call__(self, img, predicate):
        """
        Callable class method.

        :params img: image data
        :params predicate: predicate data
        """
        self.random_flip.call(img, predicate)

        outputs = self.random_flip.call_get()

        img_fliped = array_from_ptr(outputs[0],
                                    "f4",
                                    tuple(self.img_shape_np))

        return img_fliped
