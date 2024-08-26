from habana_frameworks.mediapipe.backend.nodes import OpNode
from habana_frameworks.mediapipe.backend.nodes import ComplexOpNode
from abc import abstractmethod
import numpy as np


class media_layout:
    NHWC = 0   # interleaved
    NCHW = 1   # planar
    NDHWC = 2   # interleaved
    NDCHW = 3   # planar
    MAX = 4
    idx = [None]*MAX
    str = [None]*MAX
    enum = {}
    idx[NHWC] = np.array([0, 1, 2, 3])  # for interleaved
    idx[NCHW] = np.array([1, 2, 0, 3])  # for planar
    str[NHWC] = 'CWHN'
    str[NCHW] = 'WHCN'
    idx[NDHWC] = np.array([0, 1, 2, 3, 4])  # for interleaved
    idx[NDCHW] = np.array([1, 2, 0, 3, 4])  # for planar
    str[NDHWC] = 'CWHN'  # ToDo add D in layout string
    str[NDCHW] = 'WHCN'
    enum['CWHN'] = 0
    enum['WHCN'] = 1


# following are abtract classes which needs to be derived and implemented

# this node is used to give dummy imputs useful in case of variable inputs ops
class MediaDummyNode(OpNode):
    """
    Class defining media dummy node. Node which can be used where nodes are required as place holder and doesnt serve any functional need.

    """

    def __init__(self, name, guid, device, inputs, params, cparams, node_attr, fw_params):
        """
        Constructor method.

        :params name: node name.
        :params guid: guid of node.
        :params device: device on which this node should execute.
        :params params: node specific params.
        :params cparams: backend params.
        :params node_attr: node output information
        """
        super().__init__(
            name, guid, 'cpu', inputs, params, cparams, node_attr)

# base processing node, operators derived from this node are capable of working both on cpu and hpu
# based on device specified


class MediaProcNode(OpNode):
    """
    Class defining media processor node.

    """

    def __init__(self, name, guid, device, inputs, params, cparams, node_attr, fw_params):
        """
        Constructor method.

        :params name: node name.
        :params guid: guid of node.
        :params device: device on which this node should execute.
        :params params: node specific params.
        :params cparams: backend params.
        :params node_attr: node output information
        """
        super().__init__(
            name, guid, device, inputs, params, cparams, node_attr)

    @abstractmethod
    def __call__(self):
        """
        Abstract callable class method.

        """
        pass


# hpus ops are derived form this node
class MediaHPUNode(MediaProcNode):
    """
    Class defining media HPU operator node.

    """

    def __init__(self, name, guid, device, inputs, params, cparams, node_attr, fw_params):
        """
        Constructor method.

        :params name: node name.
        :params guid: guid of node.
        :params device: device on which this node should execute.
        :params params: node specific params.
        :params cparams: backend params.
        :params node_attr: node output information
        """
        super().__init__(
            name, guid, 'hpu', inputs, params, cparams, node_attr, fw_params)

    def __call__(self):
        """
         Callable class method.

        """
        pass

# cpu ops are dervied from this node


class MediaCPUNode(MediaProcNode):
    """
    Class defining media CPU operator node.

    """

    def __init__(self, name, guid, device, inputs, params, cparams, node_attr, fw_params):
        """
        Constructor method.

        :params name: node name.
        :params guid: guid of node.
        :params guid: device on which this node should execute.
        :params params: node specific params.
        :params cparams: backend params.
        :params node_attr: node output information
        """
        super().__init__(name, guid, 'cpu', inputs, params, cparams, node_attr, fw_params)

    @abstractmethod
    def __call__(self):
        """
        Abstract callable class method.

        """
        pass


# nodes which are constants in garph are dervied from this node


class MediaConstantNode(OpNode):
    """
    Class defining media Constant node.

    """

    def __init__(self, name, guid, device, inputs, params, cparams, node_attr, fw_params):
        """
        Constructor method.

        :params name: node name.
        :params guid: guid of node.
        :params guid: device on which this node should execute.
        :params params: node specific params.
        :params cparams: backend params.
        :params node_attr: node output information
        """
        super().__init__(
            name, guid, 'cpu', inputs, params, cparams, node_attr)

    @abstractmethod
    def __call__(self):
        """
        Abstract callable class method.

        """
        pass

    @abstractmethod
    def gen_output_info(self):
        """
        Abstract  gen_output_info class method.

        """
        pass

# nodes which inputs new data in each iterations are derived from this node


class MediaFuncDataNode(OpNode):
    """
    Class defining media function node.

    """

    def __init__(self, name, guid, device, inputs, params, cparams, node_attr, fw_params):
        """
        Constructor method.

        :params name: node name.
        :params guid: guid of node.
        :params guid: device on which this node should execute.
        :params params: node specific params.
        :params cparams: backend params.
        :params node_attr: node output information
        """
        super().__init__(
            name, guid, 'cpu', inputs, params, cparams, node_attr)

    @abstractmethod
    def __call__(self):
        """
        Abstract callable class method.

        """
        pass

# reader node which is the driver of the entire graph


class MediaReaderNode(OpNode):
    """
    Class defining media reader operator node.

    """

    def __init__(self, name, guid, device, inputs, params, cparams, node_attr, fw_params):
        """
        Constructor method.

        :params name: node name.
        :params guid: guid of node.
        :params guid: device on which this node should execute.
        :params params: node specific params.
        :params cparams: backend params.
        :params node_attr: node output information
        """
        super().__init__(
            name, guid, 'cpu', inputs, params, cparams, node_attr)

    @abstractmethod
    def __iter__(self):
        """
        Abstract method to initialize iterator.

        """
        pass

    @abstractmethod
    def __next__(self):
        """
        Abstract method to get one batch of dataset ouput from iterator.

        """
        pass

    @abstractmethod
    def __len__(self):
        """
        Abstract method to get dataset length.

        returns: length of dataset in units of batch_size.
        """
        pass

    @abstractmethod
    def get_largest_file(self):
        """
        Abstract method to get largest media in the dataset.

        returns: length of dataset in units of batch_size.
        """
        pass

    @abstractmethod
    def get_media_output_type(self):
        pass


class MediaDecoderNode(OpNode):
    """
    Class defining media decoder operator node.

    """

    def __init__(self, name, guid, device, inputs, params, cparams, node_attr, fw_params):
        """
        Constructor method.

        :params name: node name.
        :params guid: guid of node.
        :params device: device on which this node should execute.
        :params params: node specific params.
        :params cparams: backend params.
        :params node_attr: node output information
        """
        super().__init__(
            name, guid, 'hpu', inputs, params, cparams, node_attr)

    @abstractmethod
    def __call__(self):
        """
        Abstract callable class method.

        """
        pass


class CPPNode(MediaCPUNode):
    """
    Class defining media compute node. Compute node does further processing of metadata
    or input buffers in media pipe backend.

    """

    def __init__(self, name, guid, device, inputs, params, cparams, node_attr, fw_params):
        """
        Constructor method.

        :params name: node name.
        :params guid: guid of node.
        :params guid: device on which this node should execute.
        :params params: node specific params.
        :params cparams: backend params.
        :params node_attr: node output information
        """
        super().__init__(
            name, guid, 'cpu', inputs, params, cparams, node_attr, fw_params)

    @abstractmethod
    def __call__(self):
        """
        Abstract callable class method.

        """
        pass

    @abstractmethod
    def add_to_pipeline(self, pipe_manager):
        """
        Abstract class to add a metadata processor node.

        """
        pass


class MediaComplexNode(ComplexOpNode):
    """
    Class defining media decoder operator node.

    """

    def __init__(self, name, device, node_attr):
        """
        Constructor method.

        :params name: node name.
        :params device: device on which this node should execute.
        :params node_attr: node output information
        """
        super().__init__(name, device, node_attr)

    @abstractmethod
    def __call__(self):
        """
        Abstract callable class method.

        """
        pass
