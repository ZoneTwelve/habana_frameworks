from enum import Enum
from media_pipe_api import MetadataOps

empty_params = {}

ssd_metadata_in_keys = ["ids", "sizes", "boxes", "labels", "lengths", "flip"]

ssd_metadata_params = {
    'workers': 2,
    'serialize': [MetadataOps.crop, MetadataOps.flip, MetadataOps.encode],
    'cropping_iterations': 1,
    'seed': -1
}
