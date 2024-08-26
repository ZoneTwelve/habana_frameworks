# this file is responsible for graph handling of mediapipe
from habana_frameworks.mediapipe.backend.cal import media_manager, pipe_manager
from habana_frameworks.mediapipe.backend.cal import graph_handler
from habana_frameworks.mediapipe.backend.nodes import construct_bidirectional_graph
from habana_frameworks.mediapipe.backend.nodes import pipe_fw_params
from habana_frameworks.mediapipe.operators.media_nodes import MediaConstantNode, MediaDummyNode
from habana_frameworks.mediapipe.operators.media_nodes import MediaReaderNode
from habana_frameworks.mediapipe.operators.media_nodes import MediaFuncDataNode
from habana_frameworks.mediapipe.operators.media_nodes import MediaDecoderNode
from habana_frameworks.mediapipe.operators.media_nodes import MediaHPUNode
from habana_frameworks.mediapipe.operators.media_nodes import MediaCPUNode
from habana_frameworks.mediapipe.operators.media_nodes import CPPNode
from habana_frameworks.mediapipe.backend.tracing import media_tracer, tracer
from habana_frameworks.mediapipe.media_types import mediaDeviceType as mdt
import copy


class graph_processor(object):
    """
    Class defining compile time processing of media nodes.

    """

    def __init__(self, device_type, output_tensors, fw_type, proxy):
        """
        Constructor method.

        """
        self._device_type_ = device_type
        self._output_tensors = output_tensors
        self._fw_type_ = fw_type
        self._proxy_ = proxy
        self._ops = []
        self._readers_ = []
        self._const_inputs_ = []
        self._decoder_ops_ = []
        self._func_inputs_ = []
        self._cpu_ops_ = []
        self._transfer_ops_ = []
        self._hpu_ops_ = []
        self._dummy_ops_ = []
        self._cpp_nodes_ = []
        self._ngops_output_tensors_ = None
        self._is_processed_ = False
        self._is_segmented_ = False
        self._hpu_graph_ = None
        self._hpu_tensor_info_ = None
        self._hpu_to_py_output_map_ = None
        self._gh_ = None

        tensors = self._output_tensors.copy()
        for t in tensors:
            t.dst_op.append(None)
        self._ops = construct_bidirectional_graph(tensors, [])

    def create_opnodes(self, batch_size, queue_depth, num_threads):
        fw_params = pipe_fw_params(
            self._device_type_, batch_size, queue_depth, num_threads)
        for i in range(len(self._ops)):
            # inplace replacement of interim opnode
            self._ops[i].create_opnode(fw_params)

        tensors = self._output_tensors.copy()
        self._ops = construct_bidirectional_graph(tensors, [])
        for i, op in enumerate(self._ops):
            op.name = op.name + "_" + str(i)
            op.counter = i
            for j, ot in enumerate(op.output_tensors):
                ot.name = op.name + "_" + 'o' + str(j)

    def process_and_validate_graph(self, batch_size, queue_depth, num_threads):
        """
        Method to process and validate graph node.

        """
        if (self._is_processed_ == True):
            return
        self._batch_size_ = batch_size
        self._queue_depth_ = queue_depth
        self._num_threads_ = num_threads
        self._is_processed_ = True

    def segment_graph(self):
        """
        Method to segment graph.

        """
        if (self._is_segmented_ == True):
            return
        for o in self._ops:
            if isinstance(o, MediaReaderNode):
                self._readers_.append(o)
            elif isinstance(o, MediaConstantNode):
                self._const_inputs_.append(o)
            elif isinstance(o, MediaFuncDataNode):
                self._func_inputs_.append(o)
            elif isinstance(o, MediaDecoderNode):
                self._decoder_ops_.append(o)
            elif isinstance(o, MediaCPUNode):
                self._cpu_ops_.append(o)
            elif isinstance(o, MediaHPUNode):
                self._hpu_ops_.append(o)
            elif isinstance(o, MediaDummyNode):
                self._dummy_ops_.append(o)
            else:
                raise RuntimeError("invalid operator")
        if (self._device_type_ == mdt.CPU):
            if (len(self._decoder_ops_) != 0 or len(self._hpu_ops_) != 0):
                raise RuntimeError(
                    "cpu pipe cannot have hpu nodes in it, use \"mixed\" mode ")
            for o in self._output_tensors:
                if (o.attr.dma_down == True):
                    raise RuntimeError(
                        "cpu pipe cannot have dma to hpu in it, use \"mixed\" mode ")
        # we currently support reader -> cpu -> hpu -> output only
        # lets check if graph contains same
        for op in self._cpu_ops_:
            out_tensors = op.output_tensors
            for o in out_tensors:
                for d in o.dst_op:
                    if (not ((d == None) or isinstance(d, MediaHPUNode)
                             or isinstance(d, MediaCPUNode)
                             or isinstance(d, MediaDecoderNode)
                             or isinstance(d, MediaFuncDataNode))):
                        raise ValueError(
                            "Detect CPU and {} mix up".format(d.__class__.__name__))

        for op in self._hpu_ops_:
            for o in op.output_tensors:
                for d in o.dst_op:
                    if (not ((d == None) or isinstance(d, MediaHPUNode))):
                        raise ValueError(
                            "Detect HPU and {} mix up".format(o.__class__.__name__))

        self._is_segmented_ = True

    def compile(self):
        """
        Method to compile graph.

        """
        self._gh_ = graph_handler(self._batch_size_,
                                  self._const_inputs_,
                                  self._func_inputs_,
                                  self._readers_,
                                  self._cpu_ops_,
                                  self._decoder_ops_,
                                  self._hpu_ops_,
                                  self._dummy_ops_,
                                  self._output_tensors,
                                  self._fw_type_,
                                  self._proxy_)
        self._gh_.compile(self._device_type_,
                          self._queue_depth_, self._num_threads_)

    def process_recipe(self):
        """
        Getter method to get graph recipe.

        """
        pass

    def get_recipe(self):
        """
        Getter method to get graph recipe.

        """
        return self._gh_.get_recipe()

    def get_num_batches(self):
        """
        Getter method to get list of media reader nodes.

        """
        return self._gh_.get_num_batches()

    def __del__(self):
        if (self._gh_ is not None):
            del self._gh_
            self._gh_ = None


class graph_executor(object):
    """
    Class defining runtime time processing of media nodes.

    """

    def __init__(self,
                 graph_processor,
                 queue_depth,
                 batch_size,
                 fw_type,
                 proxy,
                 python_proxy):
        """
        Constructor method.

        """
        self._gh_ = graph_processor._gh_

    def __del__(self):
        del self._gh_

    def start_worker(self):
        """
        Method to start backend worker.

        """
        self._gh_.start_worker()

    def stop_worker(self):
        """
        Method to stop backend worker.

        """
        pass

    def acquire_device(self, device):
        """
        Method to acquire device.

        """
        pass

    def release_device(self, ):
        """
        Method to release device.

        """
        pass

    def initialize_memory(self):
        """
        Method to initialize all backend memory.

        """
        pass

    def free_memory(self):
        """
        Method to free all backend memory.

        """
        pass

    def flush_pipeline(self):
        """
        Method to flush pending command in pipe.

        """
        pass

    # below are the executors to vall to get execution of nodes
    def initialize_iter_pipeline(self, repeat_count):
        """
        Method to initialize iterator of the pipe.

        """
        t = tracer("initialize_iter_pipeline")
        self.iterator = iter(self._gh_)

    def execute_iter_pipeline(self):
        """
        Method to execute iterator.

        """
        pass

    def execute_const_pipeline(self):
        """
        Method to execute constant pipeline.

        """
        pass

    def execute_pipeline(self):
        """
        Method to execute E2E pipeline.

        """
        pass

    def get_output(self):
        t = tracer("get_output")
        return next(self.iterator)
