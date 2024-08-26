from habana_frameworks.mediapipe.media_types import ftype as ft
from habana_frameworks.mediapipe.media_types import randomCropType as rct
from habana_frameworks.mediapipe.media_types import decoderStage as ds

# INFO: Here we will give params and its default arguments order doesnt matter
# INFO: if any parameter is not set here it will be set to zero
generic_in1_key = ["input"]
gaussian_blur_in_keys = ["images", "sigma"]
video_decoder_in_keys = [
    "input", "frame_offset", "resample_idx", "random_crop"]

gaussian_blur_params = {
    'max_sigma': 0,
    'min_sigma': 0,
    'shape': [1, 1, 1, 1, 1],  # [W,H,D,C,N]
}


reduce_params = {
    "reductionDimension": [0]
}

video_decoder_params = {
    'output_format': 'rgb-i',
    'resize': [0, 0],  # for width, height
    'crop_after_resize': [0, 0, 0, 0],  # [x, y, width, height]
    'resampling_mode': ft.BI_LINEAR,
    'random_crop_type': rct.NO_RANDOM_CROP,
    'decoder_stage': ds.ENABLE_ALL_STAGES,
    'frames_per_clip': 1,
    'max_frame_vid': 1,  # max frames with fps resampling
    'dpb_size': 16  # extra frames needed for decoder
}


random_flip_in_keys = ["input", "predicate"]

random_flip_params = {
    "horizontal": 0,
    "vertical": 0,
    "depthwise": 0
}
