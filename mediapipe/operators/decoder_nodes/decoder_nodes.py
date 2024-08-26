from habana_frameworks.mediapipe.backend.nodes import opnode_tensor_info
from habana_frameworks.mediapipe.operators.media_nodes import MediaDecoderNode
from habana_frameworks.mediapipe.operators.media_nodes import media_layout
from habana_frameworks.mediapipe.media_types import dtype as dt
from habana_frameworks.mediapipe.media_types import ftype as ft
from habana_frameworks.mediapipe.media_types import decoderType as dect

import numpy as np
import os


class image_decoder(MediaDecoderNode):
    """
    Class defining media decoder node.

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
        self.batch_size = 1
        self.output_format = params['output_format']
        self.resize = params['resize']
        self.crop_after_resize = params['crop_after_resize']
        self.dec_img_out = np.array([3, 0, 0, 0])  # channel, height , width
        self.dec_layout = media_layout.str[media_layout.NCHW]
        self.dec_params = {}
        self.dec_params["decoder_type"] = dect.IMAGE_DECODER
        # self.dec_params["is_gather_nd"] = False

        if ((params['crop_after_resize'][2] == 0) or (params['crop_after_resize'][3] == 0)):
            # Width
            self.dec_img_out[1] = self.resize[0]
            # Height
            self.dec_img_out[2] = self.resize[1]
            self.dec_params["is_crop_after_resize"] = False
        else:
            # Width
            self.dec_img_out[1] = self.crop_after_resize[2]
            # Height
            self.dec_img_out[2] = self.crop_after_resize[3]
            self.dec_params["is_crop_after_resize"] = True

        if (self.output_format == "rgb-i"):
            self.dec_layout = media_layout.NHWC
        elif (self.output_format == "rgb-p"):
            self.dec_layout = media_layout.NCHW
        else:
            raise RuntimeError("invalid layout for image decoder")
        # print("MediaDecoder layout",self.dec_layout) # TODO: check if print is needed
        # print(media_layout.idx[self.dec_layout])
        self.dec_img_out = self.dec_img_out[media_layout.idx[self.dec_layout]]
        self.dec_layout = media_layout.str[self.dec_layout]
        self.out_tensor_info = opnode_tensor_info(dt.UINT8, np.array(
            self.dec_img_out, dtype=np.uint32), self.dec_layout)
        self.dec_img_out[3] = fw_params.batch_size
        self.out_tensor_info.shape[3] = fw_params.batch_size

    def gen_output_info(self):
        """
        Method to generate output type information.

        :returns : output tensor information of type "opnode_tensor_info".
        """
        return self.out_tensor_info

    def __call__(self):
        """
        Callable class method.

        """
        pass

    def get_dec_params(self):
        return self.dec_params


class _video_decoder(MediaDecoderNode):
    """
    Class defining media decoder node.

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
        self.batch_size = 1
        self.output_format = params['output_format']
        self.resize = params['resize']
        self.crop_after_resize = params['crop_after_resize']
        # channel , width , height, numFrame
        self.dec_img_out = np.array([3, 0, 0, 0])
        self.dec_layout = media_layout.str[media_layout.NCHW]
        self.dec_params = {}
        self.dec_params["decoder_type"] = dect.VIDEO_DECODER
        self.dec_params["is_gather_nd"] = True

        self.num_container = 20
        vid_container = int(os.getenv('VID_NUM_CONTAINER', -1))
        if vid_container != -1:
            self.num_container = vid_container

        if ((params['crop_after_resize'][2] == 0) or (params['crop_after_resize'][3] == 0)):
            # Width
            self.dec_img_out[1] = self.resize[0]
            # Height
            self.dec_img_out[2] = self.resize[1]
            self.dec_params["is_crop_after_resize"] = False
            self.max_frame_vid = params['max_frame_vid']
            self.dpb_size = params['dpb_size']
        else:
            # Width
            self.dec_img_out[1] = self.crop_after_resize[2]
            # Height
            self.dec_img_out[2] = self.crop_after_resize[3]
            self.dec_params["is_crop_after_resize"] = True
            # ignore max_frame_vid, dpb_size params in case of crop_after_resize
            self.max_frame_vid = 1
            self.dpb_size = 0

        if (self.output_format == "rgb-i"):
            self.dec_layout = media_layout.NHWC
        elif (self.output_format == "rgb-p"):
            self.dec_layout = media_layout.NCHW
        else:
            raise RuntimeError("invalid layout for video decoder")

        if (params['resampling_mode'] == ft.LANCZOS) or (params['resampling_mode'] == ft.BOX):
            raise RuntimeError("Unsupported resampling_mode for video decoder")

        self.dec_img_out = self.dec_img_out[media_layout.idx[self.dec_layout]]
        self.dec_layout = media_layout.str[self.dec_layout]

        num_output_frames = fw_params.queue_depth * \
            fw_params.batch_size * params['frames_per_clip']

        if self.dec_params["is_crop_after_resize"] == False:
            num_output_frames += (self.num_container *
                                  (self.max_frame_vid + self.dpb_size))

        self.dec_img_out[3] = num_output_frames
        self.dec_params["num_output_frames"] = num_output_frames
        self.dec_params["max_frame_vid"] = self.max_frame_vid
        self.dec_params["dpb_size"] = self.dpb_size

        self.out_tensor_info = opnode_tensor_info(dt.UINT8, np.array(
            self.dec_img_out, dtype=np.uint32), self.dec_layout)

        if self.dec_params["is_crop_after_resize"] == False:
            print("video_decoder: max_frame_vid {} dpb_size {}".format(
                self.max_frame_vid, self.dpb_size))

        # print("video_decoder: num_output_frames ", num_output_frames)

    def gen_output_info(self):
        """
        Method to generate output type information.

        :returns : output tensor information of type "opnode_tensor_info".
        """
        return self.out_tensor_info

    def __call__(self):
        """
        Callable class method.

        """
        pass

    def get_dec_params(self):
        return self.dec_params
