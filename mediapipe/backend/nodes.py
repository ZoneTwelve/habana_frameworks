from habana_frameworks.mediapipe.backend.utils import get_numpy_dtype, is_valid_dtype, is_valid_layout, get_media_dtype
from abc import ABC, abstractmethod
import numpy as np
import copy


def gen_output_tensor_name(opname, port_num):
    return opname + "o" + str(port_num)


def gen_dynamic_output_tensor_name(opname, port_num):
    return opname + "do" + str(port_num)

def construct_bidirectional_graph(tensors, in_tensors):
    _ops_ = []
    # extract list of nodes used in graph
    while tensors:
        tensor_node = tensors.pop(0)
        if (not isinstance(tensor_node, TensorNode)):
            raise ValueError("Got {} instead of TensorNode\n {}".format(
                type(tensor_node), vars(tensor_node)))
        op_node = tensor_node.src_op
        # let's make it a graph that can be traversed in both ways
        for it in op_node.input_tensors:
            if op_node not in it.dst_op:
                it.dst_op.append(op_node)
        if op_node is None:
            RuntimeError("node without source")
        if op_node not in _ops_:
            _ops_.append(op_node)
        else:
            _ops_.remove(op_node)
            _ops_.append(op_node)
        for it in op_node.get_input_tensors():
            if it not in in_tensors:
                tensors.append(it)
    # since graph was constructed bottom up reverse it
    _ops_.reverse()
    return _ops_

# tensor node


class TensorAttr():
    def __init__(self, dma_down):
        self.dma_down = dma_down


class TensorNode(object):
    """
    Class defining media tensor.

    """

    def __init__(self, name, device, src_port=0, src_op=None):
        """
        Constructor method.

        :params name: tensor name.
        :params src_port: output port number of source node.
        :params src_op: source op node.
        """
        self.name = name
        if (device != 'hpu' and device != 'cpu'):
            raise RuntimeError("invalid device name ", device)
        self.device = device
        self.src_op = src_op
        self.dst_op = []
        self.src_port = src_port
        self.__data__ = None
        self.shape = None
        self.dtype = None
        self.np_shape = None
        self.np_dtype = None
        self.size = 0
        self.layout = ""
        self._pending_read_tokens_ = 0  # token based system for safe memory handling
        self.c_t = None
        self.attr = TensorAttr(False)

    def as_hpu(self):
        if (self.device == 'cpu'):
            self.attr.dma_down = True
        else:
            raise ValueError("as Hpu can be called on cpu tensor only")

    def data_write(self, data):
        """
        Method to write data to tensor node.

        :params data: data to be written to tensornode.
        """
        if (self._pending_read_tokens_ != 0):
            raise RuntimeError("output of {}[{}] written before its consumed".format(
                self.src_op.name, self.src_port))
        self.__data__ = data
        self._pending_read_tokens_ = len(self.dst_op)

    def data_read(self):
        """
        Method to read data to tensor node.

        :returns : data present in tensornode.
        """
        if (self._pending_read_tokens_ == 0):
            raise RuntimeError("output of {}[{}] read more then its produced".format(
                self.src_op.name, self.src_port))
        self._pending_read_tokens_ = self._pending_read_tokens_ - 1
        return self.__data__

    def check_data_avaiable(self):
        """
        Method to check data is avaiable to be read from tensor node.

        :params data: data to be written to tensornode.
        """
        if (self._pending_read_tokens_ > 0):
            return True
        else:
            return False

    def clear_data(self):
        """
        Method to clear data in tensor node.

        """
        self._pending_read_tokens_ = 0
        self.__data__ = None


class OpNode(ABC):
    """
    Abstract class defining media operator node.

    """

    def __init__(self, name, guid, device, input_tensors, params, cparams, node_attr):
        """
        Constructor method.

        :params name: node name.
        :params guid: guid of node.
        :params guid: device on which this node should execute.
        :params params: node specific params.
        :params cparams: backend params.
        :params node_attr: node output information
        """
        self.counter = 0
        if (device != 'hpu' and device != 'cpu'):
            raise RuntimeError("invalid device name ", device)
        self.device = device
        # self._name = "__"+type(op).__name__+"_"+str(self._counter)
        # underscore not supported
        self.name = name
        self.opname = name
        self.output_tensors = []
        for it in input_tensors:
            if (self.device == 'cpu'):
                if (it.device == 'hpu'):
                    ValueError("Input to cpu op cannot be hpu tensor")
            # else:
            # if(it.device == 'cpu'):
            #    it.dma_down = True

        self.input_tensors = input_tensors
        self.params = params
        self.cparams = cparams
        self.guid = guid
        self.node_attr = node_attr
        self.batch_size = 1
        self.c_n = None

    def gen_output_tensors(self, num_output_tensors):
        """
        Method to generate output tensors for media node.

        :params num_output_tensors: number of output tensors of the node.
        """
        for i in range(num_output_tensors):
            output_tensor_name = self.name
            if num_output_tensors > 1:
                output_tensor_name = gen_output_tensor_name(self.name,
                                                            i)
            n = TensorNode(output_tensor_name, self.device, i, self)
            self.output_tensors.append(n)

    def populate_output_tensors(self, output_tensors):
        """
        Method to populate output tensors for media node.

        :params output_tensors: list of output tensors of the node.
        """
        for o in output_tensors:
            self.output_tensors.append(o)

    def _add_dynamic_outputs_(self, op, num_dyn_out):
        """
        Method to dynamically add output tensors for media node.

        :params op: opnode.
        :params num_dyn_out: number of dynamic output tensors of the node.
        """
        num_new_outputs = num_dyn_out
        num_curr_outputs = len(op.output_tensors)
        for i in range(num_curr_outputs, num_curr_outputs+num_new_outputs):
            output_tensor_name = gen_dynamic_output_tensor_name(self.name, i)
            n = TensorNode(output_tensor_name, i, self)
            self.output_tensors.append(n)

    def get_output_tensors(self):
        """
        Getter method to get output tensors for media node.

        :returns : output tensor of the node.
        """
        if len(self.output_tensors) == 1:
            return self.output_tensors[0]
        else:
            return self.output_tensors

    def get_input_tensors(self):
        """
        Getter method to get input tensors for media node.

        returns: list of input tensors of the node.
        """
        return self.input_tensors

    def fetch_input_buffers(self):
        """
        Method to fetch input buffers for media node.

        returns: list of input buffers of the node.
        """
        inputs = []
        for i in self.input_tensors:
            # inputs.append(i.__data__.pop(0))
            inputs.append(i.data_read())
        return inputs

    def push_output_buffers(self, out_buffers):
        """
        Method to put output buffers for media node.

        :params out_buffers: list of output buffers of the node.
        """
        t = type(out_buffers)
        if t is tuple or t is list:
            for i in range(len(out_buffers)):
                self.output_tensors[i].data_write(out_buffers[i])
        else:
            self.output_tensors[0].data_write(out_buffers)

    def generate_node_info(self):
        """
        Method to generate node information.

        """
        self.out_tensor_info = self.gen_output_info()
        if (self.out_tensor_info == None):
            raise ValueError("out info of node {} is None".format(self.opname))
        if (not isinstance(self.out_tensor_info, list)):
            self.out_tensor_info = [self.out_tensor_info]
        if (len(self.out_tensor_info) != len(self.output_tensors)):
            raise ValueError(
                "out info incomplete for node {}".format(self.opname))
        for i in range(len(self.out_tensor_info)):
            o = self.out_tensor_info[i]
            if (not isinstance(o, opnode_tensor_info)):
                raise ValueError(
                    "operator {}  return output info is not opnode_tensor_info type".format(self.opname))
            if (not isinstance(o.shape, np.ndarray)):
                raise ValueError(
                    "operator {} port {} return shape is not numpy array".format(self.opname, i))
            if (not (np.uint32 == o.shape.dtype)):
                raise ValueError(
                    "operator {} port {} return shape is not uint32 dtype".format(self.opname, i))
            if (len(o.shape) > 5):
                raise ValueError(
                    "output shape from {} port {} is greater then 5".format(self.opname, i))
            if (len(o.shape) < 1):
                raise ValueError(
                    "output shape from {} port {} is less then one".format(self.opname, i))
            if (not is_valid_dtype(o.dtype)):
                raise ValueError(
                    "operator {} port {} return dtype is not in media_type.dtype".format(self.opname, i))
            if (not is_valid_layout(o.layout)):
                raise ValueError(
                    "operator {} port {} return layout is not in media_type.layout".format(self.opname, i))
            self.node_attr[i]['outputType'] = self.out_tensor_info[i].dtype
            self.output_tensors[i].shape = self.out_tensor_info[i].shape
            self.output_tensors[i].np_shape = self.out_tensor_info[i].shape[::-1]
            self.output_tensors[i].dtype = self.out_tensor_info[i].dtype
            self.output_tensors[i].np_dtype = get_numpy_dtype(
                self.out_tensor_info[i].dtype)
            self.output_tensors[i].layout = self.out_tensor_info[i].layout
            self.output_tensors[i].size = 1
            for j in range(len(self.output_tensors[i].shape)):
                self.output_tensors[i].size = self.output_tensors[i].size * \
                    self.output_tensors[i].shape[j]
            if (self.node_attr[i]['outputType'] != ''):
                self.node_attr[i]['outputType'] = get_media_dtype(
                    self.node_attr[i]['outputType'])

    @abstractmethod
    def gen_output_info(self):
        """
        Abstract method to generate output type information.

        :returns : output tensor information of type "media_ext_reader_op_tensor_info".
        """
        pass


class pipe_fw_params(object):
    """
    Class defining pipe frame work params.

    """

    def __init__(self, device, batch_size, queue_depth, num_threads):
        """
        Constructor method.

        :params batch_size: Batch size.
        """
        self.batch_size = batch_size
        self.queue_depth = queue_depth
        self.num_threads = num_threads
        self.device = device


class opnode_tensor_info(object):
    """
    Class defining return numpy tensor information of external cpu op class.

    """

    def __init__(self, dtype, shape, layout):
        """
        Constructor method.

        :params dtype: output data type.
        :params shape: output shape.
        :params layout: output layout.
        """
        self.dtype = dtype
        self.shape = shape
        self.layout = layout


class ComplexOpNode(ABC):
    def __init__(self, name, device, node_attr):
        """
        Constructor method.

        :params name: node name.
        :params node_attr: node output information
        """
        self.name = name
        self.device = device
        self.node_attr = node_attr

    @abstractmethod
    def __call__(self):
        pass


class InterimOpNode(OpNode):
    """
    Class defining media dummy node.

    """

    def __init__(self, opclass, is_complex, name, guid, device, inputs, params, cparams, node_attr):
        """
        Constructor method.

        :params name: node name.
        :params guid: guid of node.
        :params guid: device on which this node should execute.
        :params params: node specific params.
        :params cparams: backend params.
        :params node_attr: node output information
        """
        super().__init__(name, guid, device, inputs,
                         params, cparams, node_attr)
        self.__op_class = opclass
        self.is_complex = is_complex

    def create_opnode(self, fw_params):
        # there could be same op instantiated multiple times in those cases if op
        # is changing params it would affect other instance of the node
        params = copy.deepcopy(self.params)
        if (self.is_complex == False):
            op = self.__op_class(self.opname, self.guid,
                                 self.device, self.input_tensors, params, self.cparams,
                                 self.node_attr, fw_params)
            # replace name and counter with original once
            op.name = self.name
            op.counter = self.counter
            for it in op.input_tensors:
                it.dst_op = [op if do == self else do for do in it.dst_op]

            op.output_tensors = self.output_tensors
            for ot in op.output_tensors:
                ot.src_op = op
        else:
            op = self.__op_class(self.opname, self.device,
                                 params, self.node_attr, fw_params)
            for it in self.input_tensors:
                it.dst_op.remove(self)

            outputs = op(*self.input_tensors)
            if (isinstance(outputs, tuple)):
                outputs = list(outputs)
            if (not isinstance(outputs, list)):
                outputs = [outputs]
            if (len(outputs) != len(self.output_tensors)):
                raise ValueError("Mismatch in output count for {} -> gen {}!= exp {}".format(
                    self.opname, len(outputs), len(self.output_tensors)))
            for i in range(len(outputs)):
                self.output_tensors[i].attr.dma_down = self.output_tensors[i].attr.dma_down or outputs[i].attr.dma_down
                self.output_tensors[i].src_op = outputs[i].src_op
                self.output_tensors[i].src_op.output_tensors = [
                    self.output_tensors[i] if ot == outputs[i] else ot for ot in self.output_tensors[i].src_op.output_tensors]
            interim_ops = construct_bidirectional_graph(
                outputs, self.input_tensors)

            for op in interim_ops:
                op.create_opnode(fw_params)


    def gen_output_info(self):
        """
        Method to generate output type information.

        :returns : output tensor information of type "opnode_tensor_info".
        """
        pass

    def __call__(self):
        """
        Callable class method.

        """
        pass
