from habana_frameworks.mediapipe.media_types import ftype as ft
from habana_frameworks.mediapipe.media_types import randomCropType as rct
from habana_frameworks.mediapipe.media_types import decoderStage as ds

# INFO: Here we will give params and its default arguments order doesnt matter
# INFO: if any parameter is not set here it will be set to zero

image_decoder_in_keys = ["input", "random_crop"]
video_decoder_in_keys_ = ["input", "frame_offset", "resample_idx", "random_crop"]

image_decoder_params = {
    'output_format': 'rgb-i',
    'resize': [0, 0],  # for width, height
    'crop_after_resize': [0, 0, 0, 0],  # [x, y, width, height]
    'resampling_mode': ft.BI_LINEAR,
    'random_crop_type': rct.NO_RANDOM_CROP,
    'scale_min': 0,
    'scale_max': 0,
    'ratio_min': 0,
    'ratio_max': 0,
    'decoder_stage': ds.ENABLE_ALL_STAGES,
    'seed': 0
}

video_decoder_params_ = {
    'output_format': 'rgb-i',
    'resize': [0, 0],  # for width, height
    'crop_after_resize': [0, 0, 0, 0],  # [x, y, width, height]
    'resampling_mode': ft.BI_LINEAR,
    'random_crop_type': rct.NO_RANDOM_CROP,
    'decoder_stage': ds.ENABLE_ALL_STAGES,
    'frames_per_clip': 1,
    'max_frame_vid': 1,
    'dpb_size': 16
}
