from habana_frameworks.mediapipe.media_types import cropWindowType as rct
generic_in0_keys = []
generic_in1_keys = ["input"]
generic_in2_keys = ["input0", "input1"]
generic_in4_keys = ["input0", "input1", "input2", "input3"]
generic_in10_keys = ["input0", "input1", "input2", "input3", "input4",
                     "input5", "input6", "input7", "input8", "input9"]

empty_params = {}

media_constant_params = {
    'data': None,
    'layout': '',
    'shape': [1],
    'batch_broadcast': True
    # batch_broadcast: True, data and shape will have WHDC/C/WHC  (no batch dimension)
    # batch_broadcast: False, data and shape will have WHDCN/CN/WHCN (full batch dimension)
}

media_func_params = {
    'func': None,
    'shape': [1],
    'seed': 0,
    # 'unique_number': 0, this will be populated by framework can be used with seed to get unique seed
    'priv_params': {}  # user defined params can be passed here
}

media_ext_cpu_op_params = {
    'impl': None,
    'seed': 0,
    'shape': [1],
    # 'unique_number': 0, this will be populated by framework can be used with seed to get unique seed
    'priv_params': {}  # user defined params can be passed here
}

random_biased_crop_params = {
    'patch_size': [0, 0, 0],  # i.e [fcd, y, z], do not share batch_size
    'over_sampling': 0.33,
    'num_channels': 1,
    'seed': 0,
    'num_workers': 1,
    'cache_bboxes': False,
    'cache_bboxes_at_first_run': False
}

basic_crop_params = {
    'patch_size': [0, 0, 0],  # do not share batch_size
    'num_channels': 1,
    'center_crop': False
}

zoom_in_keys = ["input0", "input1", ""]
zoom_params = {
    'patch_size': [0, 0, 0],  # do not share batch_size
    'num_channels': 1
}


random_flip_in_keys__ = ["input", "predicate"]
random_flip_cpu_params__ = {
    'horizontal': 0,
    'vertical': 0,
    'depthwise': 0
}

random_bernoulli_in_keys = ["probability"]
random_bernoulli_params = {
    'seed': 100
}

ssd_crop_window_gen_keys = ["sizes", "boxes", "labels", "length"]
ssd_crop_window_gen_params = {
    'num_iterations': 1,
    'min_width': 0.3,
    'max_width': 1.0,
    'min_height': 0.3,
    'max_height': 1.0,
    'seed': 123
}
ssd_bbox_flip_keys = ["isFlip", "boxes", "length"]
ssd_encode_keys = ["boxes", "labels", "length"]

modulo_params = {
    'divisor': 360.0
}

gaussian_filter_in_keys = ["sigma"]
gaussian_filter_params = {
    'min_sigma': 0.5,
    'max_sigma': 1.5,
    'channels': 0,
    'input_depth': 0
}

random_uniform_in_keys = ["seed"]
random_uniform_params = {
    'seed': 0,
    'low': 0,
    'high': 1,
    'dims': 0,
    'shape': [0, 0, 0, 0, 0]
}

random_normal_in_keys = ["stddev", "input", "seed"]
random_normal_params = {
    'seed': 0,
    'mean': 0,
    'stddev': 0,
    "dims": 0,
    "shape": [0, 0, 0, 0, 0]
}

constant_params = {
    'constant': 0
}

where_in_keys = ["perdicate", "input0", "input1"]

# this is duplicate params which needs to be removed once kernels get fixed
crop_params = {
    'crop_w': 100,
    'crop_h': 100,
    'crop_d': 0,
    'crop_pos_x': 0.,
    'crop_pos_y': 0.,
    'crop_pos_z': 0,
    'pad_val': 0,
}

crop_window_keys = ["media"]

crop_window_params = {
    'type': rct.NO_CROP,
    'scale_min': 0.0,
    'scale_max': 1.0,
    'ratio_min': 0.0,
    'ratio_max': 1.0,
    'resize_width': 64,
    'resize_height': 64,
    'seed': -1,

}

reshape_params = {
    'size': [0, 0, 0, 0, 0],
    'tensorDim': 5,
    'layout': ''
}
