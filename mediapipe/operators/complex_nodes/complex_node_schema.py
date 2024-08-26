from habana_frameworks.mediapipe.media_types import dtype as dt
from habana_frameworks.mediapipe.backend.operator_specs import complex_schema
from habana_frameworks.mediapipe.operators.complex_nodes.gaussian_blur import gaussian_blur
from habana_frameworks.mediapipe.operators.complex_nodes.reduction import reduce_min, reduce_max
from habana_frameworks.mediapipe.operators.complex_nodes.video_decoder import video_decoder
from habana_frameworks.mediapipe.operators.complex_nodes.complex_node_params import *
from habana_frameworks.mediapipe.operators.complex_nodes.random_flip import random_flip

import media_pipe_params as mpp  # NOQA
import media_pipe_nodes as mpn  # NOQA


# add operators to the list of supported ops
# complex_schema.add_operator(oprator_name,guid, min_inputs,max_inputs,num_outputs,params_of_operator)

complex_schema.add_operator("GaussianBlur", "", 2, 2, gaussian_blur_in_keys, 1,
                            gaussian_blur_params, None, gaussian_blur, dt.FLOAT32)

complex_schema.add_operator("ReduceMin", "", 1, 2, generic_in1_key, 2,
                            reduce_params, None, reduce_min, [dt.UINT8, dt.INT32])

complex_schema.add_operator("ReduceMax", "", 1, 2, generic_in1_key, 2,
                            reduce_params, None, reduce_max, [dt.UINT8, dt.INT32])

complex_schema.add_operator("VideoDecoder", None, 2, 4, video_decoder_in_keys, 1,
                            video_decoder_params, None, video_decoder, dt.UINT8)

complex_schema.add_operator("RandomFlip", "random_flip", 2, 2, random_flip_in_keys,
                            1, random_flip_params, mpp.flipParams, random_flip, dt.NDT)

complex_schema.add_operator_cpu("RandomFlip", "random_flip", 2, 2, random_flip_in_keys,
                                1, random_flip_params, mpp.flipParams, random_flip, dt.NDT)
