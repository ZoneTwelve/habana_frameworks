from habana_frameworks.mediapipe.backend.nodes import opnode_tensor_info
from habana_frameworks.mediapipe.operators.media_nodes import MediaReaderNode
from abc import ABC, abstractmethod
import inspect


class media_ext_reader_op(MediaReaderNode):
    """
    Class defining media external reader node.

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
            name, None, device, inputs, params, cparams, node_attr, fw_params)
        self.params = params
        if (params['impl'] == None):
            raise ValueError("implementation of ext reader op not present")
        spec = inspect.getargspec(params['impl'])
        if (len(spec.args) != 3):
            msg = "{} constructor must take two arguments".format(
                str(params['impl']))
            raise RuntimeError(msg)
        self.impl_obj = params['impl'](params,fw_params)
        if (not isinstance(self.impl_obj, media_ext_reader_op_impl)):
            print(isinstance(self.impl_obj, media_ext_reader_op_impl))
            raise ValueError(
                "Tensor node function must be of type TensorFunctionNode")

    def gen_output_info(self):
        """
        Method to generate output type information.

        :returns : output tensor information of type "opnode_tensor_info".
        """
        out_info = self.impl_obj.gen_output_info()
        if (out_info == None):
            raise ValueError("out info of node {} is None".format(self.opname))
        if (not isinstance(out_info, list)):
            out_info = [out_info]
        if (len(out_info) != len(self.output_tensors)):
            raise ValueError(
                "out info incomplete for node {}".format(self.opname))
        output_info = []
        for o in out_info:
            if (not isinstance(o, media_ext_reader_op_tensor_info)):
                raise ValueError(
                    "operator {}  return output info is not opnode_tensor_info type".format(self.opname))
            oti = opnode_tensor_info(o.dtype, o.shape, o.layout)
            output_info.append(oti)
        return output_info

    def __iter__(self, *argv):
        """
        Method to initialize iterator.

        """
        self.impl_obj_iter = iter(self.impl_obj)
        return self.impl_obj_iter

    def __next__(self):
        """
        Method to get one batch of dataset ouput from iterator.

        """
        next(self.impl_obj_iter)

    def __len__(self):
        """
        Method to get dataset length.

        returns: length of dataset in units of batch_size.
        """
        return len(self.impl_obj)

    def get_largest_file(self):
        """
        Abstract method to get largest media in the dataset.

        returns: largest media element in the dataset.
        """
        return self.impl_obj.get_largest_file()

    def get_media_output_type(self):
        return self.impl_obj.get_media_output_type()


class media_ext_reader_op_impl(ABC):
    """
    Abstract class representing external reader node.

    """
    @abstractmethod
    def __init__(self, params, fw_params):
        """
        Abstract constructor method.

        :params params: private params of this node
        """
        pass

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
    def gen_output_info(self):
        """
        Abstract method to generate output type information.

        :returns : output tensor information of type "media_ext_reader_op_tensor_info".
        """
        pass

    @abstractmethod
    def get_largest_file(self):
        """
        Abstract method to get largest media in the dataset.

        """
        pass

    @abstractmethod
    def get_media_output_type(self):
        pass


class media_ext_reader_op_params(object):
    """
    Class defining param information sent to external reader op class.

    """

    def __init__(self, batch_size):
        """
        Constructor method.

        :params batch_size: Batch size.
        """
        self.batch_size = batch_size


class media_ext_reader_op_tensor_info(object):
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
