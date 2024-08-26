# this file is responsible for c/c++ abtraction from the mediapipe
from habana_frameworks.mediapipe.backend.tensor import HPUTensor
from habana_frameworks.mediapipe.backend.tensor_cpu import TensorPacker
from habana_frameworks.mediapipe.backend.nodes import TensorNode
from habana_frameworks.mediapipe.backend.nodes import gen_output_tensor_name
from habana_frameworks.mediapipe.backend.utils import get_media_dtype
from habana_frameworks.mediapipe.media_types import readerOutType as ro
from habana_frameworks.mediapipe.media_types import randomCropType as rct  # NOQA
from habana_frameworks.mediapipe.media_types import decoderStage as ds
from habana_frameworks.mediapipe.media_types import decoderType as dect
from habana_frameworks.mediapipe.operators.hpu_nodes.hpu_nodes import *
# helper function
from habana_frameworks.mediapipe.backend.legacy import legacy  # NOQA

import media_pipe_types as mpt  # NOQA
import media_pipe_params as mpp  # NOQA
import media_pipe_api as mpa  # NOQA
import media_pipe_proxy as mppy  # NOQA
import media_pipe_nodes as mpn  # NOQA
import numpy as np
import ctypes as ct
import sys
import copy


def is_ok(func_name, status):
    """
    Method to check backend return status.

    :params func_name: function name as string.
    :params status: return status of backend code.
    :raises RuntimeError: if status is failure.
    """
    if (status != mpt.status.SUCCESS):
        msg = "Media API {} failed! Error Code = {}.".format(func_name, status)
        raise RuntimeError(msg)


class media_manager():
    """
    Class defining media python to c interface for graph and compile time handling.

    """

    def __init__(self, device_type):
        """
        Constructor method.

        """
        self._device_type_ = device_type
        self._num_graph_inputs_ = 0
        self._num_graph_outputs_ = 1
        # this is assumed to have one receipe only
        self._num_graph_ws_ = 1
        # self._batch_size_ = batch_size
        # getting singletom object
        self._mm_ = mpa.MediaManager()
        # initialize mediaand synapse
        self._mm_.initialize()
        self._dev_type_id_ = mpa.mediaPipeGetDeviceTypeId("HPU")
        # mpt.device.GRECO, mpt.device.GAUDI2
        if mpt.device(self._dev_type_id_) == mpt.device.NA:
            raise ValueError("Wrong device found")
        self._mg_ = mpa.MediaGraph(mpt.device(self._dev_type_id_))

        # caching input placeholder since backend cannot identify port information for same
        self.input_placeholder_names = []
        self._recipe_ = None
        self._is_compiled_ = False

    # INFO: 1) dec_output_shape is exact output ordering you want
    #           for rgb-i case it will be 3,224,224,BATCH_SIZE
    #           for rgb-p case it will be 224,224,3,BATCH_SZIE
    #       2) output_tensor_name this is same as input tensor
    #           name of graph which is using it
    def create_decoder_node(self, tensor):
        """
        Method for creating decoder instance.

        """
        ip = mpp.mediaPlaceholderParams()
        ip.shape.numDims = len(tensor.shape)
        check_size = 1
        for i in range(ip.shape.numDims):
            ip.shape.dims[i] = tensor.shape[i]
            check_size = check_size * tensor.shape[i]
        if (check_size == 0 or len(tensor.shape) == 0):
            raise ValueError("op decoder output size is zero")
        ip.output.outputScale = 1.0            # Scale and zerop 1 and 0
        ip.output.outputZp = 0
        ip.output.outputType = mpt.dType.UINT8  # decoder supports UINT8
        ip.layout = tensor.layout
        self.input_placeholder_names.append(tensor.name)
        # print("createGenericNode ", tensor.name)
        ret = self._mm_.createGenericNode(self._mg_,
                                          [],
                                          ip,
                                          "decoder_node",
                                          tensor.name)
        is_ok("createGenericNode - decoder", ret)

    # INFO: 1) dec_output_shape is exact output ordering you want
    #           for rgb-i case it will be 3,224,224,BATCH_SIZE
    #           for rgb-p case it will be 224,224,3,BATCH_SZIE
    #       2) output_tensor_name this is same as input tensor
    #           name of graph which is using it
    def create_ops_node(self, op):
        """
        Method for creating media opnode instance.

        """
        inputs = []
        for i in op.input_tensors:
            np = mpa.nodePort()
            if (i.name in self.input_placeholder_names):

                # placeholder dont possess port information
                np.nodeName = i.name
                np.nodePort = 0
            else:
                np.nodeName = i.src_op.name
                np.nodePort = i.src_port

            inputs.append(np)

        if isinstance(op, media_hpu_user_ops):
            # Handle User Defined Op -- Generic Guid
            params_cstruct = self.create_syn_node_params(
                op.params["params"], op.params["params_type"])
            cparams = self.create_params_and_populate(op,
                                                      op.params["params"],
                                                      params_cstruct,
                                                      op.params["guid"], op.params["shape"], op.node_attr[0])

            ret = self._mm_.createGenericNode(self._mg_,
                                              inputs,
                                              ct.addressof(cparams),
                                              op.guid,
                                              op.name)
        else:
            cparams = op.cparams()
            if (hasattr(cparams, "params")):
                cparams.params = self._struct_populator_(type(cparams.params),
                                                         op.params)
            # this is because backend supports only one node attribute feild
            if len(op.output_tensors) == 1:
                op.node_attr[0]["outputType"] = get_media_dtype(
                    op.node_attr[0]["outputType"])
                cparams.output = self._struct_populator_(type(cparams.output),
                                                         op.node_attr[0])
            else:
                cparams.numOutputs = len(op.output_tensors)
                output_type = type(cparams.get_output(0))
                for i in range(len(op.output_tensors)):
                    op.node_attr[i]["outputType"] = get_media_dtype(
                        op.node_attr[i]["outputType"])
                    cparams.set_output(i, self._struct_populator_(
                        output_type, op.node_attr[i]))
            # print("createGenericNode ", op.name)
            ret = self._mm_.createGenericNode(self._mg_,
                                              inputs,
                                              cparams,
                                              op.guid,
                                              op.name)
        is_ok("createGenericNode", ret)

    def get_params_fields(self, pyparams, pyparams_type):
        fields = []
        for pykey, data_type in pyparams_type.items():
            pyvalue = pyparams[pykey]
            if isinstance(pyvalue, dict):
                print("Handle inner struct")
            elif isinstance(pyvalue, list):
                size = len(pyvalue)
                fields.append((pykey, size * data_type))
            else:
                fields.append((pykey, data_type))
        return fields

    def create_syn_node_params(self, pyparams, pyparams_type):
        syn_node_params_fields = self.get_params_fields(
            pyparams, pyparams_type)
        syn_node_params_class = self.create_ctype_class('syn_node_params', ct.Structure,
                                                        syn_node_params_fields)
        return syn_node_params_class

    def create_params_and_populate(self, op, pyparams, cparams, guid, output_shape, py_node_attr):
        media_user_node_params_fields = [("outputShape", 5 * ct.c_uint64), ("outputShapeDims", ct.c_uint64),
                                         ("guid", ct.c_char_p), ("sizeOfUserNodeParams", ct.c_uint64), ("userNodeParamsPtr", ct.c_void_p)]
        media_user_node_params_class = self.create_ctype_class(
            'media_user_node_params', ct.Structure, media_user_node_params_fields)
        media_user_node_params_obj = media_user_node_params_class()

        output_shape_dims = len(output_shape)
        media_user_node_params_obj.outputShapeDims = output_shape_dims
        for i in range(0, output_shape_dims):
            media_user_node_params_obj.outputShape[i] = output_shape[i]

        syn_node_params_obj = self._struct_populator_(cparams, pyparams)
        op.syn_params = syn_node_params_obj
        c_name = bytes(guid, 'ascii')
        media_user_node_params_obj.guid = ct.c_char_p(c_name)

        media_user_node_params_obj.sizeOfUserNodeParams = ct.sizeof(
            syn_node_params_obj)
        media_user_node_params_obj.userNodeParamsPtr = ct.addressof(
            syn_node_params_obj)

        output_fields = [("outputType", ct.c_int), ("outputZp",
                                                    ct.c_double), ("outputScale", ct.c_double)]
        output_class = self.create_ctype_class(
            'output', ct.Structure, output_fields)
        output_obj = self._struct_populator_(output_class, py_node_attr)

        op_fields = [("output", output_class),
                     ("params", media_user_node_params_class)]
        op_class = self.create_ctype_class('op_class', ct.Structure,
                                           op_fields)
        op_obj = op_class()
        op_obj.output = output_obj
        op_obj.params = media_user_node_params_obj
        return op_obj

    def create_ctype_class(self, name, base, fields):
        class CtypesStruct(base):
            _fields_ = fields
            # _pack_ = pack
        CtypesStruct.__name__ = name
        return CtypesStruct

    def _struct_populator_(self, cparams_type, pyparams):
        """
        Method for populating backend structure from python dictionary.

        """
        cparams = cparams_type()

        for pykey, pyvalue in pyparams.items():
            if (hasattr(cparams, pykey)):
                if isinstance(pyparams[pykey], dict):
                    s = self._struct_populator_(
                        type(getattr(cparams, pykey)), pyparams[pykey])
                    setattr(cparams, pykey, s)
                elif (isinstance(getattr(cparams, pykey), np.ndarray) or isinstance(pyparams[pykey], list)):
                    carr = getattr(cparams, pykey)
                    if (len(pyvalue) > len(carr)):
                        msg = "{}.{} length {} , expected {}".format(type(cparams),
                                                                     pykey,
                                                                     len(pyvalue),
                                                                     len(carr))
                        raise ValueError(msg)
                    for i in range(len(pyvalue)):
                        carr[i] = pyvalue[i]
                else:
                    # get datatype and cast the value
                    t = type(getattr(cparams, pykey))
                    setattr(cparams, pykey, t(pyvalue))
            else:
                msg = "{} params has no element {}".format(
                    type(cparams), pykey)
                raise ValueError(msg)
        return cparams

    def create_placeholder_node(self, input_tensor, placeholder_type, np_data):
        """
        Method for creating media placeholde node.

        """
        cparams = mpp.mediaPlaceholderParams()
        carr = getattr(cparams.shape, "dims")
        if (len(input_tensor.shape) > len(carr)):
            msg = "{}.{} length {} , expected {}".format(type(input_tensor.shape),
                                                         "dims",
                                                         len(input_tensor.shape),
                                                         MAX_DIMENSIONS_NUM)
            raise ValueError(msg)
        check_size = 1
        for i in range(len(input_tensor.shape)):
            cparams.shape.dims[i] = input_tensor.shape[i]
            check_size = check_size * input_tensor.shape[i]
        if (check_size == 0 or len(input_tensor.shape) == 0):
            raise ValueError("op {} output size is zero".format(
                input_tensor.src_op.name))
        cparams.shape.numDims = len(input_tensor.shape)

        cparams.output = self._struct_populator_(type(cparams.output),
                                                 input_tensor.src_op.node_attr[input_tensor.src_port])
        cparams.layout = ''

        cparams.type = placeholder_type

        if (not np_data is None):
            cparams.dataPtr = np_data.__array_interface__['data'][0]
        else:
            cparams.dataPtr = 0x0

        self.input_placeholder_names.append(input_tensor.name)
        # print("createPlaceHolder ", input_tensor.name)
        ret = self._mm_.createPlaceHolder([],
                                          input_tensor.name,
                                          cparams,
                                          self._mg_)
        is_ok("createGenericNode", ret)

    def create_output_placeholder_node(self, output_tensors):
        """
        Method for creating media placeholde node.

        """
        outputs = []
        np = mpa.nodePort()

        if (output_tensors.name in self.input_placeholder_names):
            # placeholder dont possess port information
            np.nodeName = output_tensors.name
        else:
            np.nodeName = output_tensors.src_op.name

        np.nodePort = output_tensors.src_port

        outputs.append(np)
        # print("createOutputPlaceHolder ", np.nodeName)
        ret = self._mm_.createOutputPlaceHolder(outputs,
                                                self._mg_)
        is_ok("createOutputPlaceHolder", ret)

    def compile(self, decoder_op, hpu_graph):
        """
        Method for compiling media graph and generating recipe.

        """
        if (self._is_compiled_ == True):
            return

        # logic to generate nodes automatically
        if (decoder_op is None):
            dec_output_tensor = TensorNode("dummy", "cpu")
        else:
            dec_output_tensor = decoder_op.output_tensors[0]
        for i in hpu_graph.input_unique_tensors:
            if (i == dec_output_tensor):
                self.create_decoder_node(i)
            else:
                self.create_placeholder_node(
                    i, mpt.mediaPlaceholderType.NON_CONST_PLACEHOLDER, None)

        for c in hpu_graph.const_unique_tensors:
            self.create_placeholder_node(
                c, mpt.mediaPlaceholderType.CONST_PLACEHOLDER, c.src_op())
        for op in hpu_graph.ops:
            self.create_ops_node(op)
        for o in hpu_graph.output_unique_tensors:
            self.create_output_placeholder_node(o)

        self._recipe_ = self._mm_.graphCompile(self._mg_)
        self._is_compiled_ = True

    def get_recipe(self):
        """
        Getter method to get media recipe.

        """

        if (self._is_compiled_ == True):
            return self._recipe_
        else:
            return None

    def get_hpu_tensor_info(self, recipe, num_outputs, num_inputs):
        """
        Getter method to get list of input output tensors from receipe.

        """
        tensor_info = mpa.getRecipeTensorInfo(
            recipe, num_outputs, num_inputs, 1)
        tensor_nodes = []
        for t in tensor_info:
            node = TensorNode(t.name, "hpu")
            node.shape = t.dims[:t.numDims]
            node.dtype = None
            node.np_shape = node.shape[::-1]
            node.np_dtype = None
            tensor_nodes.append(node)
        return tensor_nodes[:num_inputs], tensor_nodes[num_inputs:num_inputs+num_outputs]

    def __del__(self):
        """
        Destructor method.

        """
        # free any resources being used
        # ret = self._mm_.destroy()
        # is_ok("mm destroy", ret)
        pass  # TODO: check if _mm_.destroy() needed


#  this class is responsible for all runtime relatedmantaience in mediapipe
class pipe_manager():
    """
    Class defining media python to c interface for runtime handling.

    """

    def __init__(self, queue_depth, batch_size, framework, proxy,
                 python_proxy, ngops_output_tensors, output_tensors, hpu_to_py_output_map):
        """
        Constructor method.

        """
        self._batch_size_ = batch_size
        self._rt_host_buf_tensor_node_ = []
        self._num_rt_host_bufs_ = 0
        self._cold_run_ = True
        self._queue_depth_ = queue_depth
        self._py_proxy_ = python_proxy
        self._fw_type_ = framework
        self.tensor_list = []
        self._graph_output_tensors_ = output_tensors
        self.get_ngops_buf_funcs = []
        self.get_var_buf_funcs = []
        self.reader_output_type = ro.FILE_LIST
        self._hpu_to_py_output_map_ = hpu_to_py_output_map

        dma_nhgop_list = self._get_dma_nhgop_list_(ngops_output_tensors)

        self._pm_ = mpa.pipeManager(queue_depth,
                                    batch_size,
                                    framework,
                                    proxy,
                                    dma_nhgop_list)
        # this is only for run_hpu ctypes case
        self._pm_addr_ = self._pm_.get_ptr()

    def _get_dma_nhgop_list_(self, dma_ngop_bufs):
        """
        Method for generating list of no graph operation nodes.

        """
        nhgop_list = []
        for i in range(len(dma_ngop_bufs)):
            o = dma_ngop_bufs[i]
            metaDtype = mpa.metadata()
            metaDtype.dtype = get_media_dtype(o.dtype)
            metaDtype.numShapeDim = len(o.shape)
            check_size = 1
            for j in range(metaDtype.numShapeDim):
                metaDtype.shape[j] = o.shape[j]
                check_size = check_size * o.shape[j]
            if (check_size == 0 or metaDtype.numShapeDim == 0):
                raise ValueError("metadata {} has zero size!!!".format(i))
            nhgop_list.append(metaDtype)
        return nhgop_list

    def _gen_media_output_tensors_(self, outputs):
        """
        Method for generating list of output tensor nodes.

        """
        output_tensor = []
        for o in outputs:
            output_tensor.append(
                Tensor(o.shape, o.dtype, o.dtype))
        return output_tensor

    def acquire_device(self, device):
        """
        Method to acquire device

        """
        self.dev_type_id = mpa.mediaPipeGetDeviceTypeId("HPU")
        ret = self._pm_.acquireDevice(mpt.device(self.dev_type_id))
        is_ok("acquireDevice", ret)

    def add_cpp_compute_node(self, metadata_processor):
        """
        Method to add a cpp compute node
        """
        self._pm_.add_cpp_compute_node(metadata_processor)

    def media_init(self, recipe, num_outputs, num_inputs):
        """
        Method to initialize media backend.

        """
        if (recipe == None):
            raise RuntimeError("receipe not found")
        # this will be default present since slice and reshape are always present
        # INFO: ws is always one as of now
        num_ws = 1
        ret = self._pm_.mediaMemAlloc(recipe,
                                      num_outputs,
                                      num_inputs,
                                      num_ws)
        is_ok("mediaMemAlloc", ret)

    def decoder_init(self, decoder_op, reader_op):
        """
        Method to initialize decoder.

        """
        self.decoder_op = decoder_op
        params = decoder_op.params
        largest_file = reader_op.get_largest_file()
        in_pic_fmt = mpt.MediaPictureFormat.MEDIA_IN_NV12
        out_pic_fmt = mpt.MediaPictureFormat.MEDIA_OUT_RGB_INTERLEAVED
        if (params['output_format'] == 'rgb-i'):
            out_pic_fmt = mpt.MediaPictureFormat.MEDIA_OUT_RGB_INTERLEAVED
        elif (params['output_format'] == 'rgb-p'):
            out_pic_fmt = mpt.MediaPictureFormat.MEDIA_OUT_RGB_PLANAR
        else:
            msg = "Media decoder pic format supported rgb-i rgb-p sent :{}.".format(
                out_pic_fmt)
            raise ValueError(msg)

        self.reader_output_type = reader_op.get_media_output_type()
        if (self.reader_output_type == ro.FILE_LIST):
            input_format = mpt.mediaInputType.FILE_LIST
        elif (self.reader_output_type == ro.BUFFER_LIST):
            input_format = mpt.mediaInputType.BUFFER_LIST
        elif (self.reader_output_type == ro.ADDRESS_LIST):
            input_format = mpt.mediaInputType.ADDRESS_LIST
        else:
            raise ValueError("invalid input format")

        random_crop_type = mpt.RandomCropType.NO_RANDOM_CROP
        if (params['random_crop_type'] == rct.NO_RANDOM_CROP):
            random_crop_type = mpt.RandomCropType.NO_RANDOM_CROP
        elif (params['random_crop_type'] == rct.RANDOMIZED_AREA_AND_ASPECT_RATIO_CROP):
            random_crop_type = mpt.RandomCropType.RANDOMIZED_AREA_AND_ASPECT_RATIO_CROP
        elif (params['random_crop_type'] == rct.RANDOMIZED_ASPECT_RATIO_CROP):
            random_crop_type = mpt.RandomCropType.RANDOMIZED_ASPECT_RATIO_CROP
        elif (params['random_crop_type'] == rct.CENTER_CROP):
            random_crop_type = mpt.RandomCropType.CENTER_CROP
        else:
            raise RuntimeError("wrong random crop type provided")

        decoder_type = mpt.DecoderType.IMAGE_DECODER
        random_crop_param = mpa.mediaRandomCrop()
        max_frame_vid = 1
        frames_per_clip = 1
        is_gather_nd_for_decode = False
        num_op_frames = 1
        dpb_size = 0

        if (self.decoder_op.get_dec_params()["decoder_type"] == dect.IMAGE_DECODER):
            decoder_type = mpt.DecoderType.IMAGE_DECODER

            random_crop_param.scale_min = params['scale_min']
            random_crop_param.scale_max = params['scale_max']
            random_crop_param.ratio_min = params['ratio_min']
            random_crop_param.ratio_max = params['ratio_max']
            random_crop_param.seed = params['seed']

            max_frame_vid = 1
            frames_per_clip = 1
            is_gather_nd_for_decode = False
            num_op_frames = 1
            dpb_size = 0

        elif (self.decoder_op.get_dec_params()["decoder_type"] == dect.VIDEO_DECODER):

            if params['random_crop_type'] != rct.NO_RANDOM_CROP:
                raise RuntimeError(
                    "unsupported crop type provided for video decoder")

            decoder_type = mpt.DecoderType.VIDEO_DECODER

            random_crop_param.scale_min = 0
            random_crop_param.scale_max = 0
            random_crop_param.ratio_min = 0
            random_crop_param.ratio_max = 0
            random_crop_param.seed = 0

            max_frame_vid = self.decoder_op.get_dec_params()["max_frame_vid"]
            frames_per_clip = params['frames_per_clip']
            is_gather_nd_for_decode = self.decoder_op.get_dec_params()[
                "is_gather_nd"]
            num_op_frames = self.decoder_op.get_dec_params()[
                "num_output_frames"]
            dpb_size = self.decoder_op.get_dec_params()["dpb_size"]
        else:
            raise RuntimeError("wrong decoder type")

        decoder_stage = mpt.DecoderStage.ENABLE_ALL_STAGES
        if (params['decoder_stage'] == ds.ENABLE_ALL_STAGES):
            decoder_stage = mpt.DecoderStage.ENABLE_ALL_STAGES
        elif (params['decoder_stage'] == ds.ENABLE_SINGLE_STAGE):
            decoder_stage = mpt.DecoderStage.ENABLE_SINGLE_STAGE
        elif (params['decoder_stage'] == ds.ENABLE_TWO_STAGES):
            decoder_stage = mpt.DecoderStage.ENABLE_TWO_STAGES
        else:
            raise RuntimeError("wrong decoder stage type provided")

        crop_after_resize = []
        crop_after_resize.append(params['crop_after_resize'][0])
        crop_after_resize.append(params['crop_after_resize'][1])
        crop_after_resize.append(params['crop_after_resize'][2])
        crop_after_resize.append(params['crop_after_resize'][3])

        ret = self._pm_.decoderMemInit(largest_file,
                                       input_format,
                                       in_pic_fmt,
                                       out_pic_fmt,
                                       params['resize'][0],
                                       params['resize'][1],
                                       params['resampling_mode'],
                                       decoder_stage,
                                       random_crop_type,
                                       random_crop_param,
                                       decoder_type,
                                       max_frame_vid,
                                       frames_per_clip,
                                       crop_after_resize,
                                       is_gather_nd_for_decode,
                                       num_op_frames,
                                       dpb_size)
        is_ok("decoderMemInit", ret)

    def get_ordered_hpu_input_output_tensor_names(self):
        """
        Getter method to get list if input output tensors from receipe.

        """
        in_tensor_name = self._pm_.getInputTensorNames()
        out_tensor_name = self._pm_.getOutputTensorNames()
        return in_tensor_name, out_tensor_name

    def initialize_host_buffer(self):
        """
        Method to initialize host buffers for runtime.

        """
        # rt_host_buf_idx = np.array(rt_host_buf_idx)
        ret = self._pm_.initializeHostBuffer()
        is_ok("initializeHostBuffer", ret)

    def start_worker(self):
        """
        Method to start c worker.

        """
        ret = self._pm_.startPipelineExecutor()
        is_ok("startPipelineExecutor", ret)

    def init_iterator(self):
        """
        Method initialize iterator.

        """
        self.legacy = legacy(self._batch_size_, self._queue_depth_)

    def run_hpu(self, input_list, rand_crp, is_crp_after_resize, frame_offset, resample_idx, var_np_buf, ngop_np_buf, is_gather):
        """
        Method for performing execution on device.

        """
        output_buf = []

        if self._fw_type_ == mppy.fwType.PYTHON_FW:
            p = self._py_proxy_
            for i in range(len(self._graph_output_tensors_)):
                o = self._graph_output_tensors_[self._hpu_to_py_output_map_[i]]
                tensor_m = p.new_tensor_dataptr(
                    shape=o.np_shape, dtype=get_media_dtype(o.dtype))
                output_buf.append(tensor_m)
                self.tensor_list.append(tensor_m)
        self.legacy.run_hpu(self._pm_addr_,
                            input_list,
                            self.reader_output_type,
                            rand_crp,
                            is_crp_after_resize,
                            frame_offset,
                            resample_idx,
                            var_np_buf,
                            ngop_np_buf,
                            output_buf,
                            is_gather)

    def free_device_tensor(self, dev_addr):
        """
        Method to free device tensors.

        """
        # if self._py_proxy_ is not None:
        if self._fw_type_ == mppy.fwType.PYTHON_FW:
            self._py_proxy_.delete_tensor(dev_addr)
        elif self._fw_type_ == mppy.fwType.SYNAPSE_FW:
            self._pm_.freeRawDevBuffer(dev_addr)
        elif self._fw_type_ == mppy.fwType.TF_FW:
            pass
        elif self._fw_type_ == mppy.fwType.PYT_FW:
            pass
        else:
            raise RuntimeError("unknown FW type ", self._fw_type_)

    def get_output(self):
        """
        Method to catch the processed output from device.

        """
        hpu_outputs = self._pm_.getOutput()
        self.legacy.get_output()  # this is required to pop from the queue of legacy module
        outputs = [hpu_outputs[i] for i in self._hpu_to_py_output_map_]
        tensorlist = []
        for i in range(len(outputs)):
            if self._fw_type_ == mppy.fwType.PYTHON_FW:
                self.tensor_list.remove(outputs[i])
            tensorlist.append(HPUTensor(self._graph_output_tensors_[i],
                              outputs[i],
                              self))

        return tensorlist if len(tensorlist) > 1 else tensorlist[0]

    def as_cpu(self, device_addr, npy_buf):
        """
        Method to perform device to host transfer.

        """
        pointer, read_only_flag = npy_buf.__array_interface__['data']
        ret = self._pm_.asCpu(device_addr, pointer, npy_buf.nbytes)
        is_ok("asCpu", ret)

    def flush_pipeline(self):
        """
        Method to flush pending command from the pipe.

        """
        ret = self._pm_.flushPipeline()
        is_ok("flushPipeline", ret)
        # if self._py_proxy_ is not None:
        if self._fw_type_ == mppy.fwType.PYTHON_FW:
            # TODO: check if tensor_list can be removed
            self._py_proxy_.flush_tensors(self.tensor_list)
            self.tensor_list.clear()

    def stop_worker(self):
        """
        Method to stop c worker.

        """
        ret = self._pm_.stopPipelineExecutor()
        is_ok("stopPipelineExecutor", ret)

    def free_host_buffer(self):
        """
        Method to free host buffers.

        """
        ret = self._pm_.freeHostBuffer()
        is_ok("freeHostBuffer", ret)

    def media_deinit(self):
        """
        Method to deinitialize media backend.

        """
        ret = self._pm_.mediaMemDealloc()
        is_ok("mediaMemDealloc", ret)

    def decoder_deinit(self):
        """
        Method to deinitialize decoder.

        """
        ret = self._pm_.decoderMemDealloc()
        is_ok("decoderMemDealloc", ret)

    def release_device(self):
        """
        Method to release device.

        """
        ret = self._pm_.releaseDevice()
        is_ok("releaseDevice", ret)


class graph_handler():
    def __create_c_py_func_nodes__(self, op, arg_params, node_type, device):
        params = {}
        mfo = mpn.MediaFuncOp()
        mfo.RegisterRun(op.run)
        params['impl'] = mfo
        params['shape'] = arg_params['shape']
        params['seed'] = arg_params['seed']
        params['dtype'] = arg_params['dtype']
        return self.__create_c_nodes__(op, params, node_type, device)

    def __create_c_nodes__(self, op, params, node_type, device):
        opnode = mpn.OpNode(op.name, node_type, device)
        if (op.guid == None):
            raise ValueError("No CPU Guid found for ", op.name)
        opnode.guid = op.guid
        typeofparams = type(opnode.params)
        cparams = op.cparams()
        if (node_type == mpn.NodeType_t.NODE_HPU):
            if (hasattr(cparams, "params")):
                cparams.params = self._struct_populator_(type(cparams.params),
                                                         params)
            if len(op.output_tensors) == 1:
                op.node_attr[0]["outputType"] = get_media_dtype(
                    op.node_attr[0]["outputType"])
                cparams.output = self._struct_populator_(type(cparams.output),
                                                         op.node_attr[0])
            else:
                cparams.numOutputs = len(op.output_tensors)
                output_type = type(cparams.get_output(0))
                for i in range(len(op.output_tensors)):
                    op.node_attr[i]["outputType"] = get_media_dtype(
                        op.node_attr[i]["outputType"])
                    cparams.set_output(i, self._struct_populator_(
                        output_type, op.node_attr[i]))

            opnode.params = typeofparams(cparams)
        else:
            opnode.params = typeofparams(self._struct_populator_(type(cparams),
                                                                 params))
        for i in range(len(op.output_tensors)):
            ot = op.output_tensors[i]
            tdevice = device
            if (ot.attr.dma_down == True):
                tdevice = mpn.Device_t.DEVICE_HPU
            t = mpn.TensorNode(ot.name, tdevice)
            t.srcPort = ot.src_port
            t.srcOp = opnode
            t.dType = get_media_dtype(op.node_attr[i]["outputType"])
            t.scale = op.node_attr[i]["outputScale"]
            t.zerop = op.node_attr[i]["outputZp"]
            ot.c_t = t
            opnode.outTensors.append(t)
        for ip in op.input_tensors:
            opnode.inTensors.append(ip.c_t)
            ip.c_t.dstOps.append(opnode)
        op.c_n = opnode

    def __create_c_dummy_nodes__(self, op, params, node_type, device):
        opnode = mpn.OpNode(op.name, node_type, device)
        if (op.guid == None):
            raise ValueError("No CPU Guid found for ", op.name)
        opnode.guid = op.guid
        if len(op.output_tensors) > 1:
            raise ValueError(
                "Dummy node cannot have more then 1 output", op.name)
        if len(op.input_tensors) > 0:
            raise ValueError(
                "Dummy node cannot have inputs", op.name)
        typeofparams = type(opnode.params)
        cparams = op.cparams()
        opnode.params = typeofparams(self._struct_populator_(type(cparams),
                                                             params))
        for i in range(len(op.output_tensors)):
            ot = op.output_tensors[i]
            tdevice = device
            t = mpn.TensorNode(ot.name, tdevice)
            t.srcPort = ot.src_port
            t.srcOp = opnode
            # t.dType = get_media_dtype(op.node_attr[i]["outputType"])
            # t.scale = op.node_attr[i]["outputScale"]
            # t.zerop = op.node_attr[i]["outputZp"]
            ot.c_t = t
            opnode.outTensors.append(t)
        op.c_n = opnode

    def _struct_populator_(self, cparams_type, pyparams):
        """
        Method for populating backend structure from python dictionary.

        """
        cparams = cparams_type()
        for pykey, pyvalue in pyparams.items():
            if (hasattr(cparams, pykey)):
                if isinstance(pyparams[pykey], dict):
                    s = self._struct_populator_(
                        type(getattr(cparams, pykey)), pyparams[pykey])
                    setattr(cparams, pykey, s)
                elif isinstance(getattr(cparams, pykey), np.ndarray):
                    carr = getattr(cparams, pykey)
                    if (len(pyvalue) > len(carr)):
                        msg = "{}.{} length {} , expected {}".format(type(cparams),
                                                                     pykey,
                                                                     len(pyvalue),
                                                                     len(carr))
                        raise ValueError(msg)
                    for i in range(len(pyvalue)):
                        carr[i] = pyvalue[i]
                elif isinstance(getattr(cparams, pykey), int) and isinstance(pyvalue, np.ndarray):
                    setattr(cparams, pykey,
                            pyvalue.__array_interface__['data'][0])
                else:
                    # get datatype and cast the value
                    t = type(getattr(cparams, pykey))
                    if t is type(None):
                        setattr(cparams, pykey, pyvalue)
                    else:
                        setattr(cparams, pykey, t(pyvalue))
            else:
                msg = "{} params has no element {}".format(
                    type(cparams), pykey)
                raise ValueError(msg)
        size = sys.getsizeof(cparams)
        return cparams

    def __init__(self, batch_size, const_nodes, func_nodes, reader_nodes, cpu_nodes, decoder_nodes, hpu_nodes, dummy_nodes, outputs, fw_type, proxy):
        self.__reader_nodes__ = reader_nodes
        self.__const_nodes__ = const_nodes
        self.__func_nodes__ = func_nodes
        self.__cpu_nodes__ = cpu_nodes
        self.__decoder_nodes__ = decoder_nodes
        self.__hpu_nodes__ = hpu_nodes
        self.__dummy_nodes__ = dummy_nodes
        self.__outputs__ = outputs
        self._batch_size_ = batch_size
        self._fw_type_ = fw_type
        self._proxy_ = proxy
        self.__c_outputs__ = []

        for r in self.__reader_nodes__:
            self.__create_c_nodes__(
                r, r.params, mpn.NodeType_t.NODE_READER, mpn.Device_t.DEVICE_CPU)
        for op in self.__const_nodes__:
            self.__create_c_nodes__(
                op, op.params, mpn.NodeType_t.NODE_CPU, mpn.Device_t.DEVICE_CPU_CONST)
        for op in self.__dummy_nodes__:
            self.__create_c_dummy_nodes__(
                op, op.params, mpn.NodeType_t.NODE_DUMMY, mpn.Device_t.DEVICE_CPU)
        for op in self.__cpu_nodes__:
            self.__create_c_nodes__(
                op, op.params, mpn.NodeType_t.NODE_CPU, mpn.Device_t.DEVICE_CPU)
        for op in self.__func_nodes__:
            self.__create_c_py_func_nodes__(
                op, op.params,  mpn.NodeType_t.NODE_CPU, mpn.Device_t.DEVICE_CPU)
        for op in self.__decoder_nodes__:
            self.__create_c_nodes__(
                op, op.params, mpn.NodeType_t.NODE_DECODER_BATCHER, mpn.Device_t.DEVICE_CPU)
        for op in self.__hpu_nodes__:
            self.__create_c_nodes__(
                op, op.params, mpn.NodeType_t.NODE_HPU, mpn.Device_t.DEVICE_HPU)
        for o in self.__outputs__:
            self.__c_outputs__.append(o.c_t)
        self.is_compiled = False

    def compile(self, device_type, queue_depth, num_threads):
        self.device_type = device_type
        if (self.is_compiled == True):
            return
        outputs = mpn.tensor_node_list()
        for o in self.__outputs__:
            outputs.append(o.c_t)
        self.gh = mpn.GraphHandler(self.device_type,
                                   self._batch_size_,
                                   queue_depth,
                                   num_threads,
                                   outputs,
                                   self._fw_type_,
                                   self._proxy_)
        self.gh.PreProcessGraph()
        self.gh.MakeCookBook()
        self.is_compiled = True

    def get_num_batches(self):
        return self.gh.Len()

    def start_worker(self):
        self.gh.AllocateControllers()

    def stop_worker(self):
        self.gh.DeallocateControllers()

    def __iter__(self):
        self.gh.Iter()
        return self

    def __next__(self):
        return TensorPacker(self.gh.GetOutput())

    def __del__(self):
        del self.gh
