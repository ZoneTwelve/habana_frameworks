from habana_frameworks.mediapipe.backend.operator_specs import schema
from habana_frameworks.mediapipe.operators.metadata_nodes.metadata_nodes import ssd_metadata_processor
from habana_frameworks.mediapipe.media_types import dtype as dt
from habana_frameworks.mediapipe.operators.metadata_nodes.metadata_node_params import *
import media_pipe_params as mpp  # NOQA

schema.add_operator("SSDMetadata", None, 5, 6, ssd_metadata_in_keys, 6,
                    ssd_metadata_params, None, ssd_metadata_processor, dt.NDT)
