from habana_frameworks.mediapipe.backend.nodes import opnode_tensor_info
from habana_frameworks.mediapipe.operators.media_nodes import CPPNode
from habana_frameworks.mediapipe.backend.utils import get_str_dtype
from habana_frameworks.mediapipe.media_types import dtype as dt
from media_pipe_api import SSDMetadataProcessor, MetadataOps
import numpy as np
import inspect


class ssd_metadata_processor(CPPNode):
    """
    Class representing ssd metadata processor node.

    """

    def __init__(self, name, guid, device, inputs, params, cparams, out_info, fw_params):
        """
        Constructor method.

        :params name: node name.
        :params guid: guid of node.
        :params guid: device on which this node should execute.
        :params params: node specific params.
        :params cparams: backend params.
        :params out_info: node output information
        """
        super().__init__(
            name, None, device, inputs, params, cparams, out_info, fw_params)
        self.params = params
        self.batch_size = 1
        self.batch_size = fw_params.batch_size

    def add_to_pipeline(self, pipe_manager):
        ops = self.params["serialize"]
        flip_random_input = ""
        if MetadataOps.flip in ops:
            flip_random_input = self.input_tensors[5].name

        self.metadata_processor = SSDMetadataProcessor(self.params["workers"],
                                                       self.params["serialize"],
                                                       flip_random_input,
                                                       self.params["cropping_iterations"],
                                                       self.params["seed"])
        pipe_manager.add_cpp_compute_node(self.metadata_processor.addr())


    def gen_output_info(self):
        """
        Method to generate output type information.

        :returns : output tensor information of type "opnode_tensor_info".
        """
        # Crop windows
        windows = opnode_tensor_info(dt.FLOAT32, np.array(
            [4, self.batch_size], dtype=np.uint32), "")

        # Image ids
        ids = opnode_tensor_info(dt.UINT32, np.array(
            [self.batch_size], dtype=np.uint32), "")

        # Image sizes
        sizes = opnode_tensor_info(dt.UINT32, np.array(
            [2, self.batch_size], dtype=np.uint32), "")

        # Image bounding boxes
        bboxes = opnode_tensor_info(dt.FLOAT32, np.array(
            [4, 8732, self.batch_size], dtype=np.uint32), "")

        # Image labels
        labels = opnode_tensor_info(dt.UINT32, np.array(
            [8732, self.batch_size], dtype=np.uint32), "")

        # Lengths
        lengths = opnode_tensor_info(dt.UINT32, np.array(
            [self.batch_size], dtype=np.uint32), "")

        output_info = [windows, ids, sizes, bboxes, labels, lengths]
        return output_info

    def __call__(self, *argv):
        """
        Callable class method.

        :params *argv: list of inputs to this node.
        """
        ids = argv[0]
        sizes = argv[1]
        input_bboxes = argv[2]
        input_labels = argv[3]
        input_lengths = argv[4]
        windows = np.empty([self.batch_size, 4], dtype=np.uint32)
        bboxes = np.zeros([self.batch_size, 8732, 4], dtype=np.float32)  # ToDo
        labels = np.zeros([self.batch_size, 8732], dtype=np.uint32)  # ToDo
        lengths = np.zeros([self.batch_size], dtype=np.uint32)  # ToDo
        for i in range(self.batch_size):
            boxes_len = input_lengths[i]
            bboxes[i, 0:boxes_len] = input_bboxes[i, 0:boxes_len]
            labels[i, 0:boxes_len] = input_labels[i, 0:boxes_len]
            lengths[i] = boxes_len

        return windows, ids, sizes, bboxes, labels, lengths
