import os
from habana_frameworks.mediapipe.media_types import dtype as dt
from habana_frameworks.mediapipe.backend.operator_specs import schema
from habana_frameworks.mediapipe.media_types import dtype as dt
from habana_frameworks.mediapipe.operators.hpu_nodes.hpu_nodes import *
from habana_frameworks.mediapipe.operators.hpu_nodes.hpu_node_params import *
import ctypes as ct
import media_pipe_params as mpp  # NOQA


# add operators to the list of supported ops
# schema.add_operator(oprator_name,guid, min_inputs,max_inputs,num_outputs,params_of_operator)
schema.add_operator("Resize", 'resize_image', 1, 1, generic_in1_key, 1,
                    resize_params, mpp.resizeParams, media_hpu_ops, dt.UINT8)

schema.add_operator("ColorSpaceConversion", 'color_space_conversion',
                    1, 1, generic_in1_key, 1, color_space_params, mpp.colorSpaceConversionParams, media_hpu_ops, dt.UINT8)

schema.add_operator("Crop", 'crop',
                    1, 1, generic_in1_key, 1, crop_params, mpp.cropParams, media_hpu_ops, dt.UINT8)

schema.add_operator("CropMirrorNorm", 'crop_mirror_norm',
                    3, 3, ["input", "mean", "inv_std"], 1, cmn_params, mpp.cropMirrorNormParams, media_hpu_ops, dt.UINT8)

schema.add_operator("Reshape", 'reshape',
                    1, 1, generic_in1_key, 1, reshape_params, mpp.mediaReshapeNdParams, media_hpu_ops, dt.UINT8)

schema.add_operator("Contrast", 'contrast',
                    1, 2, contrast_in_key, 1, contrast_params, mpp.contrastParams, media_hpu_ops, dt.UINT8)

schema.add_operator("Brightness", 'brightness',
                    1, 2, brightness_in_key, 1, brightness_params, mpp.brightnessParams, media_hpu_ops, dt.UINT8)

schema.add_operator("Saturation", 'saturation',
                    1, 2, saturation_in_key, 1, saturation_params, mpp.saturationParams, media_hpu_ops, dt.UINT8)

schema.add_operator("Transpose", 'transpose', 1, 1, generic_in1_key, 1,
                    transpose_params, mpp.mediaTransposeParams, media_hpu_ops, dt.UINT8)

schema.add_operator("Slice", 'slice', 1, 1, generic_in1_key, 1,
                    slice_params, mpp.mediaSliceParams, media_hpu_ops, dt.UINT8)

schema.add_operator("Concat", 'concat', 2, 10, generic_in2_key, 1,
                    concat_params, mpp.mediaConcatParams, media_hpu_ops, dt.UINT8)

schema.add_operator("Split", 'split', 1, 1, generic_in1_key, 10,
                    split_params, mpp.mediaSplitParams, media_hpu_ops, [dt.UINT8]*10)


schema.add_operator("Constant", 'constant', 0, 1, generic_in1_key, 1,
                    constant_params, mpp.mediaConstantParams, media_hpu_ops, dt.UINT8)

schema.add_operator("Where", 'where', 3, 3, where_in_keys, 1,
                    empty_params, mpp.mediaConstantParams, media_hpu_ops, dt.UINT8)

schema.add_operator("MemCpy", 'memcpy', 1, 1, generic_in1_key, 1,
                    empty_params, mpp.mediaCopyParams, media_hpu_ops, dt.UINT8)

schema.add_operator("Cast", 'cast', 1, 1, generic_in1_key, 1,
                    cast_params, mpp.mediaCastParams, media_hpu_ops, dt.FLOAT32)

schema.add_operator("BitwiseOr", 'bitwise_or', 2, 2, generic_in2_key, 1,
                    empty_params, mpp.bitwiseBinaryParams, media_hpu_ops, dt.UINT8)

schema.add_operator("BitwiseAnd", 'bitwise_and', 2, 2, generic_in2_key, 1,
                    empty_params, mpp.bitwiseBinaryParams, media_hpu_ops, dt.UINT8)

schema.add_operator("BitwiseXor", 'bitwise_xor', 2, 2, generic_in2_key, 1,
                    empty_params, mpp.bitwiseBinaryParams, media_hpu_ops, dt.UINT8)

schema.add_operator("GatherND", "gather_nd", 2, 2, gathernd_in_keys,
                    1, empty_params, mpp.mediaGatherParams, media_hpu_ops, dt.UINT8)

schema.add_operator("Add", 'add', 2, 2, generic_in2_key, 1,
                    empty_params, mpp.mediaArithmeticParams, media_hpu_ops, dt.FLOAT32)

schema.add_operator("Neg", 'neg', 1, 1, generic_in1_key, 1,
                    empty_params, mpp.mediaArithmeticParams, media_hpu_ops, dt.FLOAT32)  # TODO: To test

schema.add_operator("Mult", 'mult', 2, 2, generic_in2_key, 1,
                    empty_params, mpp.mediaArithmeticParams, media_hpu_ops, dt.FLOAT32)

schema.add_operator("Sub", 'sub', 2, 2, generic_in2_key, 1,
                    empty_params, mpp.mediaArithmeticParams, media_hpu_ops, dt.FLOAT32)


schema.add_operator("_random_flip_", "random_flip", 2, 2, random_flip_in_keys__,
                    1, random_flip_params__, mpp.flipParams, media_hpu_ops, dt.UINT8)

schema.add_operator("CoinFlip", "random_bernoulli", 1, 2, random_bernoulli_in_keys,
                    1, random_bernoulli_params, mpp.mediaRandomBernoulliParams, media_hpu_ops, dt.UINT8)

schema.add_operator("RandomUniform", "random_uniform", 0, 1, random_uniform_in_keys,
                    1, random_uniform_params, mpp.mediaRandomUniformParams, media_hpu_ops, dt.UINT8)

schema.add_operator("Flip", "flip_3d", 1, 1, generic_in1_key,
                    1, random_flip_params__, mpp.flipParams, media_hpu_ops, dt.UINT8)

schema.add_operator("Normalize", "normalize", 1, 1, generic_in1_key,
                    1, normalize_params, mpp.normalizeParams, media_hpu_ops, dt.UINT8)

schema.add_operator("Hue", "hue", 1, 2, hue_in_keys,
                    1, hue_params, mpp.hueParams, media_hpu_ops, dt.UINT8)

schema.add_operator("ImageRotate", 'image_rotate',
                    1, 1, generic_in1_key, 1, image_rotate_params, mpp.mediaRotateParams, media_hpu_ops, dt.UINT8)

schema.add_operator("FrameRateReduction", "frame_rate_reduction", 1, 1, generic_in1_key,
                    1, frr_params, mpp.FrameRateReductionParams, media_hpu_ops, dt.UINT8)

schema.add_operator("Pad", "pad", 1, 1, generic_in1_key,
                    1, pad_params, mpp.mediaPadParams, media_hpu_ops, dt.UINT8)

schema.add_operator("Clamp", "clamp", 1, 3, clamp_in_keys,
                    1, clamp_params, mpp.mediaClampParams, media_hpu_ops, dt.UINT8)

schema.add_operator("RandomNormal", "random_normal", 1, 3, random_normal_in_keys,
                    1, random_normal_params, mpp.mediaRandomNormalParams, media_hpu_ops, dt.UINT8)

schema.add_operator("SpatialConv", "spatial_convolution", 2, 2, conv_in_keys,
                    1, conv_params, mpp.mediaConvolutionParams, media_hpu_ops, dt.FLOAT32)

# schema.add_operator("GaussianBlur3D", "gaussian_blur_3d", 2, 2, gaussian_blur_in_keys,
#                    1, gaussian_blur_params, mpp.mediaGaussianBlurParams, media_hpu_ops, dt.FLOAT32)

schema.add_operator("_ReduceMin_", "reduce_min", 1, 1, generic_in1_key,
                    2, reduce_params__, mpp.mediaReduceParams, media_hpu_ops, [dt.UINT8, dt.INT32])

schema.add_operator("_ReduceMax_", "reduce_max", 1, 1, generic_in1_key,
                    2, reduce_params__, mpp.mediaReduceParams, media_hpu_ops, [dt.UINT8, dt.INT32])

schema.add_operator("MediaExtHpuOp", "user_defined", 1, 10, generic_in10_key,
                    2, user_defined_params, None, media_hpu_user_ops, dt.UINT8)
