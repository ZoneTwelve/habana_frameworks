from habana_frameworks.mediapipe.operators.media_nodes import MediaComplexNode
from habana_frameworks.mediapipe import fn
from habana_frameworks.mediapipe.media_types import dtype as dt
from habana_frameworks.mediapipe.operators.cpu_nodes.cpu_nodes import media_function

# import numpy as np


class gather_nd_func(media_function):
    def __init__(self, params):
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']

    def __call__(self):  # resample_idx
        #op_gather_index = np.zeros(shape=self.np_shape, dtype=self.np_dtype)
        return None


class video_decoder(MediaComplexNode):
    # Output from video_decoder:
    # rgb-i: [C, W(resize/crop),H (resize/crop), D, N]
    # rgb-p: [W(resize/crop), H (resize/crop), C, D, N]

    def __init__(self, name, device, params, node_attr, fw_params):
        """
        Constructor method.

        :params params: node specific params.
        :params node_attr: node output information
        """

        if(device == "cpu"):
            raise ValueError("CPU video decoder not supported")
        super().__init__(name, device, node_attr)
        self.device = device

        channels = 3
        self.batch_size = fw_params.batch_size
        self.queue_depth = fw_params.queue_depth
        self.max_frame_vid = params['max_frame_vid']
        self.frame_per_clip = params['frames_per_clip']
        output_format = params['output_format']

        self.decode = fn._VideoDecoder(output_format=output_format,
                                       random_crop_type=params['random_crop_type'],
                                       # width ,height
                                       resize=params['resize'],
                                       crop_after_resize=params['crop_after_resize'],
                                       resampling_mode=params['resampling_mode'],
                                       decoder_stage=params['decoder_stage'],
                                       max_frame_vid=self.max_frame_vid,
                                       frames_per_clip=self.frame_per_clip,
                                       dpb_size=params['dpb_size'])

        if((params['crop_after_resize'][2] == 0) or (params['crop_after_resize'][3] == 0)):
            width = params['resize'][0]
            height = params['resize'][1]
        else:
            width = params['crop_after_resize'][2]
            height = params['crop_after_resize'][3]

        if(output_format == "rgb-i"):
            # self.reshape_pre_gather = fn.Reshape(size=[channels, width, height, self.frame_per_clip * self.batch_size * self.queue_depth], tensorDim=4, layout='')
            self.reshape_post_gather = fn.Reshape(
                size=[channels, width, height, self.frame_per_clip, self.batch_size], tensorDim=5, layout='')
        elif(output_format == "rgb-p"):
            # self.reshape_pre_gather = fn.Reshape(size=[width, height, channels, self.frame_per_clip * self.batch_size * self.queue_depth], tensorDim=4, layout='')
            self.reshape_post_gather = fn.Reshape(
                size=[width, height, channels, self.frame_per_clip, self.batch_size], tensorDim=5, layout='')
        else:
            raise RuntimeError("invalid output format")

        self.gather_nd_indices = fn.MediaFunc(func=gather_nd_func,
                                              dtype=dt.INT32,  # ToDo: Update media_pipeline if indices dtype is updated
                                              shape=[
                                                  1, self.frame_per_clip * self.batch_size]
                                              )
        self.gather_nd_op = fn.GatherND()

        print("video_decoder: op width {} op height {} frames_per_clip {}".format(
            width, height, self.frame_per_clip))

    def __call__(self, *inputs):
        video = self.decode(*inputs)
        # video = self.reshape_pre_gather(video)
        indices = self.gather_nd_indices()
        video = self.gather_nd_op(video, indices)
        video = self.reshape_post_gather(video)
        return video
