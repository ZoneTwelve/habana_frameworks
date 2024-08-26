from habana_frameworks.mediapipe.backend.nodes import opnode_tensor_info
from habana_frameworks.mediapipe.operators.media_nodes import MediaCPUNode
from habana_frameworks.mediapipe.media_types import dtype as dt
from habana_frameworks.mediapipe.backend.utils import array_from_ptr
import media_resize as mr
import numpy as np


class zoom(MediaCPUNode):
    """
    Class representing media zoom cpu node.

    """

    def __init__(self, name, guid, device, inputs, params, cparams, node_attr, fw_params):
        """
        Constructor method.

        :params name: node name.
        :params guid: guid of node.
        :params device: device on which this node should execute.
        :params params: node specific params.
        :params cparams: backend params.
        :params node_attr: node output information
        """
        super().__init__(
            name, None, device, inputs, params, cparams, node_attr, fw_params)
        self.patch_size = params['patch_size'].copy()
        self.patch_size_np = self.patch_size[::-1]
        self.num_channels = params['num_channels']

        if len(self.patch_size) != 3:
            raise ValueError("3D patch size expected")

        self.batch_size = fw_params.batch_size
        self.zoom = mr.MediaResize(
            self.patch_size_np, self.num_channels, self.batch_size)

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
        out_info = []
        o = opnode_tensor_info(self.img_dtype, np.array(
            self.img_shape, dtype=np.uint32), "")
        out_info.append(o)
        o = opnode_tensor_info(self.lbl_dtype, np.array(
            self.lbl_shape, dtype=np.uint32), "")
        out_info.append(o)
        return out_info

    def __call__(self, img_sliced, lbl_sliced, cropped_patch):
        """
        Callable class method.

        :params img_sliced: image data
        :params lbl_sliced: label data
        :params cropped_patch: crop data
        """

        patch_size_np_ar = np.array(self.patch_size_np, dtype=dt.UINT32)
        cropped_patch_np_ar = np.flip(cropped_patch, axis=1)

        offset = np.zeros([self.batch_size, 3], dtype=dt.UINT32)

        for i in range(self.batch_size):
            # if crop/zoom to be done
            if (np.array_equal(patch_size_np_ar, cropped_patch_np_ar[i]) == False):
                offset[i] = (
                    (patch_size_np_ar - cropped_patch_np_ar[i]) + 1) // 2

        # img = PIL.Image.Resampling.BICUBIC
        # lbl = PIL.Image.Resampling.NEAREST
        outputs = self.zoom.resize(
            img_sliced, lbl_sliced, offset, cropped_patch_np_ar)

        # [batch_size] + [num_channel] + self.patch_size_np
        img_zoom_op = array_from_ptr(outputs[0],
                                     "f4",
                                     tuple(self.img_shape_np))
        lbl_zoom_op = array_from_ptr(outputs[1],
                                     "u1",
                                     tuple(self.lbl_shape_np))

        return img_zoom_op, lbl_zoom_op
