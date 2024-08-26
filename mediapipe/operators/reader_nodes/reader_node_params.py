from habana_frameworks.mediapipe.media_types import dtype as dt
from habana_frameworks.mediapipe.media_types import lastBatchStrategy as lbs
from habana_frameworks.mediapipe.media_types import clipSampler as cs

# INFO: Here we will give params and its default arguments order doesnt matter
# INFO: if any parameter is not set here it will be set to zero

generic_in0_keys = []

media_ext_reader_op_params = {
    'impl': None,
    'seed': 0,
    'priv_params': {}  # user defined params can be passed here
}

read_image_from_dir_params = {
    'dir': "/",
    'format': "jpg",  # or ["jpg", "JPG", "jpeg", "JPEG"]
    'shuffle': True,
    'seed': None,
    'max_file': None,
    'is_modulo_slice': True,
    'drop_remainder': False,
    'pad_remainder': False,
    'label_dtype': dt.UINT64,
    'num_slices': 1,
    'slice_index': 0,
    'file_list': None,
    'class_list': None,
    'file_sizes': None,
    'file_classes': None,
    'shard_once' : None, # if set to some bool value over rides shuffle_across_dataset
    'last_batch_strategy' : lbs.NONE, # this overrides drop_remainder and pad_remainder
}


read_video_from_dir_params = {
    'dir': "",
    'format': "mp4",      # updated for video
    'seed': None,
    'drop_remainder': False,
    'pad_remainder': False,
    'label_dtype': dt.UINT64,
    'num_slices': 1,
    'slice_index': 0,
    'file_list': [],
    'class_list': [],
    'file_classes': [],
    'frames_per_clip': 1,      # added for video
    'clips_per_video': 1,      # added for video for fixed_clip_mode=False
    'target_frame_rate': 0,    # added for video for fixed_clip_mode=False
    'step_between_clips': 1,   # added for video for fixed_clip_mode=False
    'sampler': cs.RANDOM_SAMPLER,  # added for video for fixed_clip_mode=False
    'fixed_clip_mode': False,  # added for video
    'start_frame_index': 0,    # added for video for fixed_clip_mode=True
    'shuffle': True            # for fixed_clip_mode=True
    # 'max_file', 'file_sizes'
}

coco_reader_params = {
    'root':         "",
    'annfile':   "",
    'drop_remainder': False,
    'pad_remainder': False,
    'num_slices': 1,
    'slice_index': 0,
    'shuffle': True,
    'max_file': None,
    'partial_batch': False,
    'seed': None,
    'shard_once' : None, # if set to some bool value over rides shuffle_across_dataset
    'last_batch_strategy' : lbs.NONE, # this overrides drop_remainder and pad_remainder
}


ssd_metadata_params = {
    'crop_iterations':  1,
    'batch_size':       1,
    'flip_probability': 0.5,
    'seed':             0,
    'dboxes':           None
}


read_numpy_from_dir_params = {
    'file_list': [],
    'dir': "",
    'pattern': "xyz_*.npz",
    'shuffle': True,
    'seed': -1,
    'max_file': "",
    'num_readers': 1,
    'drop_remainder': False,
    'pad_remainder': False,
    'num_slices': 1,
    'slice_index': 0,
    # when dataset contains same shape in all npy's dense should be set
    'dense': True,
    # when shuffle_across_dataset set to true all dataset instances should receive same seed.
    'shuffle_across_dataset': False,
    # type of slice to happen modulo slice or contigiuos slice
    'is_modulo_slice': True,
    # cache all files, best for small dataset and network filesystem
    'cache_files': False,
    'shard_once' : None, # if set to some bool value over rides shuffle_across_dataset
    'last_batch_strategy' : lbs.NONE, # this overrides drop_remainder and pad_remainder
}
