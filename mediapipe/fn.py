import sys
from habana_frameworks.mediapipe.operators.media_schema import schema as s
from habana_frameworks.mediapipe.operators.media_schema import complex_schema as cs
from habana_frameworks.mediapipe.operators.media_params import node_output_attributes
from habana_frameworks.mediapipe import fn
from habana_frameworks.mediapipe.backend.nodes import InterimOpNode


def operator_add(name: str, is_complex: bool):
    """
    Method to  add operator to fn.
    """
    class operator():
        """
        Class defining media operator.
        """

        def __init__(self, **kwargs):
            """
            Constructor method.

            """
            if ("name" in kwargs.keys()):
                self.opname = kwargs["name"]
                del kwargs["name"]
            else:
                self.opname = str(type(self).__name__)
            self.is_complex = type(self).__is_complex__
            self.device = None
            # layout is not mapped to any thing in c++ need to see how to handle it
            self.layout = ''
            self.output_scale = 1.0
            self.output_zerop = 0.0
            if ("device" in kwargs.keys()):
                self.device = kwargs["device"]
                del kwargs["device"]
            if self.is_complex == True:
                self.device, self.schema = cs.get_operator_schema(
                    type(self).__name__, self.device)
            else:
                self.device, self.schema = s.get_operator_schema(
                    type(self).__name__, self.device)
            self.num_outputs = self.schema.getNumOutputs()
            node_output_attr = node_output_attributes.copy()
            node_output_attr["outputType"] = self.schema.getDtype()
            if ("output_scale" in kwargs.keys()):
                node_output_attr["outputScale"] = kwargs["output_scale"]
                del kwargs["output_scale"]
            if ("output_zerop" in kwargs.keys()):
                node_output_attr["outputZp"] = kwargs["output_zerop"]
                del kwargs["output_zerop"]
            if ("dtype" in kwargs.keys()):
                if (isinstance(node_output_attr["outputType"], list)):
                    if (not isinstance(kwargs["dtype"], list)):
                        kwargs["dtype"] = [kwargs["dtype"]]
                    for i in range(len(kwargs["dtype"])):
                        node_output_attr["outputType"][i] = kwargs["dtype"][i]
                else:
                    node_output_attr["outputType"] = kwargs["dtype"]
                del kwargs["dtype"]
            if ("num_outputs" in kwargs.keys()):
                self.num_outputs = kwargs["num_outputs"]
                del kwargs["num_outputs"]
            self.node_attr = self.__construct_node_output_attributes__(node_output_attr,
                                                                       self.num_outputs)

            self.params = self.schema.updateparams(**kwargs)
            self.guid = self.schema.getGuid()
            self.op_class = self.schema.getOpClass()
            self.cparams = self.schema.getCParams()

        def __construct_node_output_attributes__(self, node_output_attr, num_outputs):
            for key in node_output_attr:
                if (not isinstance(node_output_attr[key], list)):
                    node_output_attr[key] = [node_output_attr[key]]
                if len(node_output_attr[key]) != num_outputs:
                    if (len(node_output_attr[key]) == 1):
                        for i in range(num_outputs - 1):
                            node_output_attr[key].append(
                                node_output_attr[key][0])
            node_output_attr_list = []
            for i in range(num_outputs):
                tmp = node_output_attributes.copy()
                for key in node_output_attr:
                    tmp[key] = node_output_attr[key][i]
                node_output_attr_list.append(tmp)
            return node_output_attr_list

        def __call__(self, *inputs, **kwargs):
            """
            Callable class method which generates the input and output nodes of the operator.

            """
            minInps = self.schema.getMinInputs()
            maxInps = self.schema.getMaxInputs()
            inputs = list(inputs)
            if (len(kwargs) > 0):
                keys = self.schema.getInputKeys()[len(inputs):]
                for key, value in kwargs.items():
                    if not key in keys:
                        raise ValueError(
                            'input {} for operator {} is invalid'.format(
                                key, self.name))
                num_dummys = 0
                for k in keys:
                    if k in kwargs.keys():
                        for i in range(num_dummys):
                            op = fn.MediaDummy()
                            inputs.append(op())
                        num_dummys = 0
                        inputs.append(kwargs[k])
                    else:
                        if (len(inputs) < minInps):
                            raise ValueError(
                                "missing mandatory input ", k)
                        else:
                            num_dummys = num_dummys + 1
            inp_len = len(inputs)
            if (inp_len < minInps or inp_len > maxInps):
                raise RuntimeError(
                    "input count mismatch, min input {} max input {} , curr input {}".format(minInps, maxInps, inp_len))
            op = InterimOpNode(self.op_class,
                               self.is_complex,
                               self.opname, self.guid,
                               self.device, inputs, self.params, self.cparams,
                               self.node_attr)
            op.gen_output_tensors(self.num_outputs)
            return op.get_output_tensors()
            """
            if self.is_complex:
                return self.complex_operator_generator(inputs)
            else:
                return self.operator_generator(inputs)"""

        def complex_operator_generator(self, inputs):
            op = self.op_class(self.opname, self.device,
                               self.params, self.node_attr)
            outputs = op(*inputs)
            if (isinstance(outputs, tuple)):
                outputs = list(outputs)
            num_outputs = 1
            if (isinstance(outputs, list)):
                num_outputs = len(outputs)

            if (num_outputs != self.num_outputs):
                raise ValueError("Mismatch in output count for {} -> gen {}!= exp {}".format(
                    self.opname, num_outputs, self.num_outputs))

            return outputs

        def operator_generator(self, inputs):
            op = InterimOpNode(self.op_class,
                               self.opname, self.guid,
                               self.device, inputs, self.params, self.cparams,
                               self.node_attr)
            op.gen_output_tensors(self.num_outputs)
            return op.get_output_tensors()

    operator.__name__ = name
    operator.__is_complex__ = is_complex
    return operator


module = sys.modules[__name__]
operators = s.get_operators_list()
for op in operators:
    op_class = operator_add(op, False)
    op_class.__module__ = module.__name__
    setattr(module, op, op_class)

operators = cs.get_operators_list()
for op in operators:
    op_class = operator_add(op, True)
    op_class.__module__ = module.__name__
    setattr(module, op, op_class)
