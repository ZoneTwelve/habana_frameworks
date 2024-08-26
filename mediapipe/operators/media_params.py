from habana_frameworks.mediapipe.media_types import dtype as dt
from habana_frameworks.mediapipe.operators.hpu_nodes.hpu_node_params import *
from habana_frameworks.mediapipe.operators.reader_nodes.reader_node_params import *
from habana_frameworks.mediapipe.operators.cpu_nodes.cpu_node_params import *
from habana_frameworks.mediapipe.operators.decoder_nodes.decoder_node_params import *


node_output_attributes = {
    'outputType': dt.UINT8,
    'outputZp': 0,
    'outputScale': 1.0, }
