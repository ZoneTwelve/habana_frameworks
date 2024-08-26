from habana_frameworks.mediapipe.operators.media_nodes import MediaReaderNode


class reader_cpu_ops_node(MediaReaderNode):
    """
    Class representing media random biased crop cpu node.

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
            name, guid, device, inputs, params, cparams, node_attr, fw_params)


    def gen_output_info(self):
        """
        Method to generate output type information.

        :returns : output tensor information of type "opnode_tensor_info".
        """
        pass

    def __iter__(self):
        """
        Callable class method.

        :params img: image data
        :params lbl: label data
        """
        pass

    def __next__(self):
        """
        Callable class method.

        :params img: image data
        :params lbl: label data
        """
        pass

    def get_largest_file(self):
        """
        Method to get largest media in the dataset.

        returns: largest media element in the dataset.
        """
        pass

    def get_media_output_type(self):
        pass

    def __len__(self):
        """
        Method to get dataset length.

        returns: length of dataset in units of batch_size.
        """
        pass
