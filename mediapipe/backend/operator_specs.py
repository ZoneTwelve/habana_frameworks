# File containing class of operator specification
import copy


class Operator(object):
    """
    Class defining the media nodes and its specifications.

    """

    def __init__(
            self,
            name,
            guid,
            min_inputs,
            max_inputs,
            input_keys,
            num_outputs,
            params,
            cparams,
            op_class,
            dtype):
        """
        Constructor method.

        :params name: node name.
        :params guid: guid of the node.
        :params min_inputs: minimun inputs required by the node.
        :params max_inputs: maximum inputs required by the node.
        :params num_outputs: number of output produced by the node.
        :params params: params dictionary for this node.
        :params cparams: backend params for this node.
        :params op_class: class to which this node belongs to.
        """
        self.name = name
        self.__min_inputs = min_inputs
        self.__max_inputs = max_inputs
        self.__num_outputs = num_outputs
        self.__input_keys = input_keys
        self.__params = params
        self.__guid = guid
        self.__cparams = cparams
        self.__op_class = op_class
        self.__dtype = dtype

    def check_existing_attributes(self, guid,
                                  params):
        if (guid != None and self.__guid != None):
            if (guid != self.__guid):
                raise ValueError(
                    "guid mismatch with existing op ", self.name)
        if (params != self.__params):
            if (len(params.keys()) < len(self.__params.keys())):
                if (params.items() > self.__params.items()):
                    raise ValueError(
                        "params mismatch with existing op ", self.name, self.__params, params)
            elif (len(params.keys()) > len(self.__params.keys())):
                if (params.items() < self.__params.items()):
                    raise ValueError(
                        "params mismatch with existing op ", self.name, self.__params, params)
            else:
                raise ValueError(
                    "params mismatch with existing op ", self.name, self.__params, params)

    def updateparams(self, **kwargs):
        """
        Method to update defualts operator params.

        :params **kwargs: dictionary of params to be updated.
        :returns : updated dictionary of params.
        """
        params = copy.deepcopy(self.__params)
        for key, value in kwargs.items():
            if key in params.keys():
                params[key] = copy.deepcopy(value)
            else:
                raise RuntimeError(
                    'param {0} for operator {1} is invalid'.format(
                        key, self.name))
        return params

    def getGuid(self):
        """
        Getter method to get guid.

        """
        return self.__guid

    def getNumInputs(self):
        """
        Getter method to get number of inputs.

        """
        return self.__min_inputs, self.__max_inputs

    def getMaxInputs(self):
        """
        Getter method to get max inputs.

        """
        return self.__max_inputs

    def getMinInputs(self):
        """
        Getter method to get min inputs.

        """
        return self.__min_inputs

    def getInputKeys(self):
        """
        Getter method to get min inputs.

        """
        return self.__input_keys

    def getNumOutputs(self):
        """
        Getter method to get num outputs.

        """
        return self.__num_outputs

    def getOpClass(self):
        """
        Getter method to get opcode class.

        """
        return self.__op_class

    def getDtype(self):
        """
        Getter method to get default dtype.

        """
        return self.__dtype

    def getCParams(self):
        """
        Getter method to get cparams.

        """
        return self.__cparams


class operator_schema():
    """
    Call defining default schema for a operator.

    """

    def __init__(self):
        """
        Constructor method.

        """
        self._ops_list = []
        self._op_to_obj = {}
        self._op_to_obj_cpu = {}

    def add_operator(self,
                     name,
                     guid,
                     min_inputs,
                     max_inputs,
                     input_keys,
                     num_outputs,
                     params,
                     cparams,
                     op_class,
                     dtype):
        """
        Method to add operators

        :params name: operator name
        :params guid: guid of operator
        :params min_inputs: minimum inputs required by operator.
        :params max_inputs: maximum inpus reuired by operator.
        :params num_outputs: numbers of outputs of operator.
        :params params: parameter of the given operator.
        :params cparams: backend parameter of the given operator.
        :params op_class: class to which this oprator belongs to.
        :params dtype: defaults output dtypes of the operator.
        """
        if name in self._ops_list:
            if (name in self._op_to_obj.keys()):
                raise ValueError("Duplicate op being added ", name)
            if not (name in self._op_to_obj_cpu.keys()):
                raise ValueError("op missing in cpu list ", name)
            op = self._op_to_obj_cpu[name]
            op.check_existing_attributes(guid, params)
        else:
            self._ops_list.append(name)
        op = Operator(name, guid, min_inputs, max_inputs,
                      input_keys, num_outputs, params, cparams, op_class, dtype)
        self._op_to_obj[name] = op

    def add_operator_cpu(self,
                         name,
                         guid,
                         min_inputs,
                         max_inputs,
                         input_keys,
                         num_outputs,
                         params,
                         cparams,
                         op_class,
                         dtype):
        """
        Method to add operators

        :params name: operator name
        :params guid: guid of operator
        :params min_inputs: minimum inputs required by operator.
        :params max_inputs: maximum inpus reuired by operator.
        :params num_outputs: numbers of outputs of operator.
        :params params: parameter of the given operator.
        :params cparams: backend parameter of the given operator.
        :params op_class: class to which this oprator belongs to.
        :params dtype: defaults output dtypes of the operator.
        """
        if name in self._ops_list:
            if (name in self._op_to_obj_cpu.keys()):
                raise ValueError("Duplicate op being added ", name)
            if not (name in self._op_to_obj.keys()):
                raise ValueError("op missing in list ", name)
            op = self._op_to_obj[name]
            op.check_existing_attributes(guid, params)
        else:
            self._ops_list.append(name)
        op = Operator(name, guid, min_inputs, max_inputs,
                      input_keys, num_outputs, params, cparams, op_class, dtype)
        self._op_to_obj_cpu[name] = op

    def get_operators_list(self):
        """
        Getter method to get operator listin schema.

        """
        return self._ops_list

    def get_operator_schema(self, operator, device):
        """
        Getter method to get full schema.

        """
        if (device is None):
            # if no device mentoned default priority given to hpu
            if (operator in self._op_to_obj.keys()):
                return "hpu", self._op_to_obj[operator]
            elif (operator in self._op_to_obj_cpu.keys()):
                return "cpu", self._op_to_obj_cpu[operator]
            else:
                raise RuntimeError(operator, " not implemented")
        elif (device == "cpu"):
            if (operator in self._op_to_obj_cpu.keys()):
                return device, self._op_to_obj_cpu[operator]
            else:
                raise RuntimeError(operator, " not implemented in cpu")
        elif (device == "hpu"):
            if (operator in self._op_to_obj.keys()):
                return device, self._op_to_obj[operator]
            else:
                raise RuntimeError(operator, " not implemented in hpu")
        else:
            raise RuntimeError(operator, " sent with invalid device ", device)


schema = operator_schema()
complex_schema = operator_schema()
