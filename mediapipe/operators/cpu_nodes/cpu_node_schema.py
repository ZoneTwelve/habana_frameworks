from habana_frameworks.mediapipe.backend.operator_specs import schema
from habana_frameworks.mediapipe.operators.cpu_nodes.cpu_nodes import media_constants, media_dummy, media_func_data, media_ext_cpu_op
from habana_frameworks.mediapipe.operators.cpu_nodes.random_biased_crop import random_biased_crop
from habana_frameworks.mediapipe.operators.cpu_nodes.zoom import zoom
from habana_frameworks.mediapipe.operators.cpu_nodes.basic_crop import basic_crop
from habana_frameworks.mediapipe.operators.cpu_nodes.gaussian_filter import gaussian_filter
from habana_frameworks.mediapipe.operators.cpu_nodes.cpu_ops_node import cpu_ops_node
from habana_frameworks.mediapipe.operators.cpu_nodes.cpu_ops_node import cpu_media_func_node
from habana_frameworks.mediapipe.media_types import dtype as dt
from habana_frameworks.mediapipe.operators.cpu_nodes.cpu_node_params import *
import media_pipe_params as mpp  # NOQA
import media_pipe_nodes as mpn  # NOQA


schema.add_operator_cpu("MediaDummy", "dummy", 0, 0, generic_in0_keys, 1,
                    empty_params, mpn.EmptyParams_t, media_dummy, None)

schema.add_operator_cpu("MediaExtCpuOp", None, 0, 10, generic_in10_keys, 1,
                        media_ext_cpu_op_params, None, media_ext_cpu_op, dt.UINT8)

schema.add_operator_cpu("Zoom", None, 3, 3, zoom_in_keys, 2,
                        zoom_params, None, zoom, dt.UINT8)

schema.add_operator_cpu("BasicCrop", None, 1, 1, generic_in1_keys, 1,
                        basic_crop_params, None, basic_crop, dt.UINT8)

schema.add_operator_cpu("RandomBiasedCrop", "random_biased_crop", 2, 2, generic_in2_keys,  3,
                        random_biased_crop_params, mpn.RBCParams_t, random_biased_crop, dt.NDT)

schema.add_operator_cpu("CoinFlip", "random_bernoulli", 1, 1, random_bernoulli_in_keys, 1,
                        random_bernoulli_params, mpn.RandomBernoulliParams_t, cpu_ops_node, dt.NDT)

schema.add_operator_cpu("RandomUniform", "random_uniform", 0, 1, random_uniform_in_keys, 1,
                        random_uniform_params, mpn.RandomUniformParams_t, cpu_ops_node, dt.NDT)

schema.add_operator_cpu("RandomNormal", "random_normal", 1, 3, random_normal_in_keys, 1,
                        random_normal_params, mpn.RandomNormalParams_t, cpu_ops_node, dt.NDT)

schema.add_operator_cpu("_random_flip_", "random_flip", 2, 2, random_flip_in_keys__, 1,
                        random_flip_cpu_params__, mpn.RandomFlipParams_t, cpu_ops_node, dt.NDT)

schema.add_operator_cpu("Mult", "mult", 2, 2, generic_in2_keys, 1,
                        empty_params, mpn.EmptyParams_t, cpu_ops_node, dt.NDT)

schema.add_operator_cpu("Where", "where", 3, 3, where_in_keys, 1,
                        empty_params, mpn.EmptyParams_t, cpu_ops_node, dt.NDT)

schema.add_operator_cpu("Add", "add", 2, 2, generic_in2_keys, 1,
                        empty_params, mpn.EmptyParams_t, cpu_ops_node, dt.NDT)

schema.add_operator_cpu("Crop", "crop", 1, 1, generic_in1_keys, 1,
                        crop_params, mpn.CropParams_t, cpu_ops_node, dt.NDT)

schema.add_operator_cpu("Constant", "constant", 0, 0, generic_in0_keys, 1,
                        constant_params, mpn.ConstantParams_t, cpu_ops_node, dt.NDT)

schema.add_operator_cpu("GaussianFilter", "gaussian_filter", 1, 1, gaussian_filter_in_keys,  1,
                        gaussian_filter_params, mpn.GaussianFilterParams_t, gaussian_filter, dt.NDT)

schema.add_operator_cpu("MediaConst", "media_const", 0, 0, generic_in0_keys, 1,
                        media_constant_params, mpn.MediaConstantParams_t, media_constants, dt.NDT)

schema.add_operator_cpu("MediaFunc", "media_func", 0, 4, generic_in4_keys, 1,
                        media_func_params, mpn.MediaFuncParams_t, media_func_data, dt.UINT8)

schema.add_operator_cpu("SSDCropWindowGen", "ssd_crop_window_gen", 4, 4, ssd_crop_window_gen_keys,  5,
                        ssd_crop_window_gen_params, mpn.SSDCropWindowGenParams_t, cpu_ops_node, dt.NDT)

schema.add_operator_cpu("SSDBBoxFlip", "ssd_bbox_flip", 3, 3, ssd_bbox_flip_keys,  1,
                        empty_params, mpn.EmptyParams_t, cpu_ops_node, dt.NDT)

schema.add_operator_cpu("SSDEncode", "ssd_encode", 3, 3, ssd_encode_keys,  2,
                        empty_params, mpn.EmptyParams_t, cpu_ops_node, dt.NDT)

schema.add_operator_cpu("Modulo", "modulo", 1, 1, generic_in1_keys,  1,
                        modulo_params, mpn.ModuloParams_t, cpu_ops_node, dt.NDT)

schema.add_operator_cpu("CropWindowGen", "crop_window_gen", 1, 1, crop_window_keys,  1,
                        crop_window_params, mpn.CropWindowGeneratorParams_t, cpu_ops_node, dt.NDT)

schema.add_operator_cpu("Reshape", "reshape",
                        1, 1, generic_in1_keys, 1, reshape_params, mpn.ReshapeParams_t, cpu_ops_node, dt.NDT)
