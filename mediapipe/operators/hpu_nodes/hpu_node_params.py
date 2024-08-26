
# INFO: Here we will give params and its default arguments order doesnt matter
# INFO: if any parameter is not set here it will be set to zero

generic_in1_key = ["input"]
generic_in2_key = ["input0", "input1"]
generic_in10_key = ["input0", "input1", "input2", "input3", "input4",
                    "input5", "input6", "input7", "input8", "input9"]

empty_params = {}

resize_params = {
    'mode': 1,
    'scaleDim1': 0,
    'scaleDim2': 0,
    'scaleDim3': 0,
    'coordTransMode': 0,
    'nearestMode': 0,
    'excludeOutside': False,
    'useScales': False,
    'cubicCoeffA': 0,
    'size1': 0,
    'size2': 0,
    'size3': 1
}

random_bernoulli_in_keys = ["probability", "seed"]

random_bernoulli_params = {
    'seed': 100
}

random_uniform_in_keys = ["seed"]

random_uniform_params = {
    'low': 0,
    'high': 1,
    'seed': 0,
    'dims': 0,
    'shape': [0, 0, 0, 0, 0]
}

# color_space_modes defined in trees/npu-stack/specs_external/perf_lib_layer_params.h
color_space_modes = {
    'RGB_TO_YCBCR': 0,
    'RGB_TO_BGR': 1,
    'YCBCR_TO_RGB': 2,
    'YCBCR_TO_BGR': 3,
    'BGR_TO_RGB': 4,
    'BGR_TO_YCBCR': 5,
    'GRAY_TO_RGB': 6,
    'GRAY_TO_BGR': 7,
    'GRAY_TO_YCBCR': 8,
    'RGB_TO_GRAY': 9,
    'YCBCR_TO_GRAY': 10,
    'BGR_TO_GRAY': 11
}
color_space_params = {
    'colorSpaceMode': color_space_modes['RGB_TO_BGR']
}

crop_params = {
    'crop_w': 100,
    'crop_h': 100,
    'crop_d': 0,
    'crop_pos_x': 0.,
    'crop_pos_y': 0.,
    'crop_pos_z': 0,
    'pad_val': 0,
}

cmn_params = {
    'mirror': 0,
    'crop_w': 100,
    'crop_h': 100,
    'crop_d': 0,
    'crop_pos_x': 0.,
    'crop_pos_y': 0.,
    'crop_pos_z': 0,
    'pad_val': 0,
}

reshape_params = {
    'size': [0, 0, 0, 0, 0],
    'tensorDim': 5,
    'layout': ''
}

contrast_in_key = ["input", "contrast"]

contrast_params = {
    'contrast_scale': 0.0
}

brightness_in_key = ["input", "brightness"]
brightness_params = {
    'brightness_scale': 0.0
}

saturation_in_key = ["input", "saturation"]
saturation_params = {
    'saturation_level': 0.0
}

transpose_params = {
    'permutation': [0, 1, 2, 3, 4],
    'tensorDim': 5}

slice_params = {
    'axes': [0, 0, 0, 0, 0],
    'starts': [0, 0, 0, 0, 0],
    'ends': [0, 0, 0, 0, 0],
    'steps': [0, 0, 0, 0, 0]
}

concat_params = {
    'axis': 0
}

split_params = {
    'axis': 0
}

cast_params = {
    'round_mode': 0
}

constant_params = {
    'constant': 0.0
}

gathernd_in_keys = ["input", "indices"]


where_in_keys = ["condition", "input0", "input1"]

# random flip node is part of complex node so please refer comples node params file
random_flip_in_keys__ = ["input", "predicate"]

random_flip_params__ = {
    "horizontal": 0,
    "vertical": 0,
    "depthwise": 0
}

normalize_params = {
    "scale": 0.0,
    "shift": 0.0,
    "axis": 0,
    "batch": False
}

hue_in_keys = ["input", "degree"]

hue_params = {
    "degree": 0
}

# ToD: To add
image_rotate_params = {
    "m_angle": 90.,
    "m_inputCenterX": 0,
    "m_inputCenterY": 0,
    "m_outputCenterX": 0,
    "m_outputCenterY": 0,
    "m_background": 0
}

frr_params = {
    "frrRatio": 1.0
}

pad_modes = {
    'PAD_MODE_CONSTANT': 0,
    'PAD_MODE_REFLECT': 1,
    'PAD_MODE_EDGE': 2,
    'PAD_MODE_SYMMETRIC': 3
}

pad_params = {
    "mode": pad_modes["PAD_MODE_CONSTANT"],
    "value": 0.0,
    "pads": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

clamp_in_keys = ["input", "lower_bound", "upper_bound"]

clamp_params = {
    "upperBound": 0.0,
    "lowerBound": 0.0
}

random_normal_in_keys = ["stddev", "input", "seed"]

random_normal_params = {
    "mean": 0.0,
    "stddev": 0.0,
    "seed": 0.0,
    "dims": 0,
    "shape": [0, 0, 0, 0, 0]
}

conv_in_keys = ["input", "kernel"]

conv_params = {
    "nGroups": 1,
    "kW": 1,
    "kH": 1,
    "dW": 1,
    "dH": 1,
    "dilW": 1,
    "dilH": 1,
    "padT": 0,
    "padB": 0,
    "padL": 0,
    "padR": 0,
}

# reduce node is part of complex node so please refer comples node params file
reduce_params__ = {
    "reductionDimension": 0
}

user_defined_params = {
    "min_inputs": 1,
    "max_inputs": 2,
    "num_outputs": 1,
    "guid": "",
    "params": None,
    "params_type": None,
    "shape": [0, 0, 0, 0, 0]
}
