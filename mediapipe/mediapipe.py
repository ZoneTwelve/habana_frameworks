# main mediapipe file : master controller
from habana_frameworks.mediapipe.backend.utils import get_media_fw_type
from habana_frameworks.mediapipe.backend.utils import getDeviceIdFromDeviceName
# from habana_frameworks.mediapipe.backend.graph import graph_executor, graph_processor
from habana_frameworks.mediapipe.operators.media_nodes import media_layout as cl
from habana_frameworks.mediapipe.backend.iterator import HPUGenericIterator as iter
from habana_frameworks.mediapipe.media_types import layout as lt
from habana_frameworks.mediapipe.media_types import mediaDeviceType as mdt
from habana_frameworks.mediapipe.backend.proxy_impl import set_c_proxy
from habana_frameworks.mediapipe.backend.logger import printf
from habana_frameworks.mediapipe.backend.tracing import media_tracer, tracer
import media_pipe_types as mpt
import media_pipe_proxy as mppy  # NOQA
from abc import ABC, abstractmethod
import numpy as np
import time


class MediaPipe(ABC):
    """
    Abstract class representing media pipe.

    """
    framework_type = None
    framework_proxy = None

    @abstractmethod
    def __init__(self, device=None, prefetch_depth=2, batch_size=1, num_threads=1, pipe_name=None):
        """
        Constructor method.

        :params device: media device to run mediapipe on. <hpu/hpu:0>
        :params prefetch_depth: mediapipe prefetch count. <1/2/3>
        :params batch_size: mediapipe output batch size.
        :params channel: mediapipe output channel.
        :params height: mediapipe output height.
        :params width: mediapipe output width.
        :params pipe_name: mediapipe user defined name.
        :params layout: mediapipe output layout. <lt.NHWC/lt.WHCN>

        """
        self._batch_size = batch_size
        self._queue_depth = prefetch_depth
        if not isinstance(num_threads, int):
            raise ValueError("num_threads must be integer")
        self._num_threads = num_threads
        self._graph_compiled = False
        self._pipeline_init = False
        self._gp = None
        self._device = device
        self._num_outstanding_cmd = 0
        self._pipename = pipe_name
        if (device == mdt.CPU or device == mdt.MIXED):
            from habana_frameworks.mediapipe.backend.graph import graph_executor, graph_processor
            self.__graph_executor = graph_executor
            self.__graph_processor = graph_processor
            self._device_type, self._device_id = device, 0
            self.__execute = self.__run_cpu__

        elif (device == mdt.LEGACY):
            from habana_frameworks.mediapipe.backend.graph_legacy import graph_executor, graph_processor
            self.__graph_executor = graph_executor
            self.__graph_processor = graph_processor
            self._device_type, self._device_id = device, 0
            self.__execute = self.__run__
        else:
            print(" Warning!!!!!! : Unsupported device please use legacy/cpu/mixed")
            print("Falling back to legacy")
            device = "legacy"
            from habana_frameworks.mediapipe.backend.graph_legacy import graph_executor, graph_processor
            self.__graph_executor = graph_executor
            self.__graph_processor = graph_processor
            self._device_type, self._device_id = device, 0
            self.__execute = self.__run__


        print("MediaPipe device {} device_type {} device_id {} pipe_name {}".format(
            device, self._device_type, self._device_id, pipe_name))
        self._fw_type = mppy.fwType.SYNAPSE_FW
        self._proxy = 0x0  # this must be 0 and not None
        self._python_proxy = None
        # INFO: as of now only 1 recipe is supported
        self._iter_repeat_count = 1
        self._cold_run = True
        self.__tracer = media_tracer()
        printf("{}  created.".format(self._pipename))

    def __del__(self):
        """
        Destructor method.

        """
        self.del_iter()
        if (self._gp is not None):
            del self._gp
            self._gp = None
        printf("closing mediapipe : ", self._pipename)

    @abstractmethod
    def definegraph(self):
        """
        Abstract method defining the media graph.
        Derived class defines the control flow of the media nodes in this method.

        """
        pass

    def setOutputShape(self, batch_size, channel, height, width, layout=lt.NHWC):
        """
        Setter method to set media pipe output shape.

        :params channel: mediapipe output channel.
        :params height: mediapipe output height.
        :params width: mediapipe output width.
        :params layout: mediapipe output layout. <lt.NHWC/lt.WHCN>
        """
        self._img_cwhb_ = np.array(
            [channel, width, height, batch_size], dtype=np.uint32)
        self._img_out_ = self._img_cwhb_[cl.idx[cl.enum[layout]]]

    def getDeviceId(self):
        """
        Getter method to get media pipe device id

        :returns : mediapipe device id.
        """
        return self._device_id

    def getBatchSize(self):
        """
        Getter method to get media pipe batch_size

        :returns : mediapipe batch_size.
        """
        return self._batch_size

    def build(self):
        """
        Method to build media pipe nodes and generate the recipe.

        """
        if (self._graph_compiled == True):
            # graph already built
            return
        self._graph_compiled = True
        # call user defined graph function
        output_tensors = self.definegraph()
        if isinstance(output_tensors, tuple):
            output_tensors = list(output_tensors)
        elif not isinstance(output_tensors, list):
            output_tensors = [output_tensors]
        self._gp = self.__graph_processor(
            self._device_type, output_tensors, self._fw_type, self._proxy)
        self._gp.create_opnodes(self._batch_size,
                                self._queue_depth,
                                self._num_threads)
        self._gp.segment_graph()
        self._gp.process_and_validate_graph(self._batch_size,
                                            self._queue_depth,
                                            self._num_threads)
        self._recipe_ = self._gp.compile()
        self._gp.process_recipe()

    def set_proxy(self, fw_type, proxy):
        """
        Setter method to set media proxy.

        :params fw_type: framework to be used by media. <SYNAPSE_FW/TF_FW/PYTHON_FW/PYT_FW>
        :params proxy : c++ proxy address.
        """
        if MediaPipe.framework_type is None:
            MediaPipe.framework_type = get_media_fw_type(fw_type)
            if not isinstance(MediaPipe.framework_type, mppy.fwType):
                raise RuntimeError(" Invalid proxy type.")
            self._fw_type = MediaPipe.framework_type
            MediaPipe.framework_proxy = proxy
            if (MediaPipe.framework_type == mppy.fwType.PYTHON_FW):
                MediaPipe.framework_c_proxy = set_c_proxy(
                    MediaPipe.framework_proxy)
                self._proxy = MediaPipe.framework_c_proxy
                self._python_proxy = MediaPipe.framework_proxy
            else:
                self._proxy = MediaPipe.framework_proxy
        else:
            fw_type = get_media_fw_type(fw_type)
            if fw_type != MediaPipe.framework_type:
                raise RuntimeError("A different framework already initialized")
            self._fw_type = MediaPipe.framework_type
            if (MediaPipe.framework_type == mppy.fwType.PYTHON_FW):
                self._proxy = MediaPipe.framework_c_proxy
                self._python_proxy = MediaPipe.framework_proxy
            else:
                self._proxy = MediaPipe.framework_proxy

    def set_repeat_count(self, count):
        """
        Setter method to set mediapipe iteration repeat count.

        :params count: number of time media pipe iteration to be repeated.
        """
        self._iter_repeat_count = count

    def iter_init(self):
        """
        Method to initialize mediapipe iterator.

        """
        t = tracer("iter_init")
        printf("{} iter init ".format(self._pipename))
        if (self._graph_compiled != True):
            raise RuntimeError("Pipe build() not called!!!!")
        if (self._pipeline_init == False):
            self._gexe_ = self.__graph_executor(self._gp,
                                                self._queue_depth,
                                                self._batch_size,
                                                self._fw_type,
                                                self._proxy,
                                                self._python_proxy)
            self._gexe_.acquire_device(self._device_type)
            self._gexe_.initialize_memory()
            self._gexe_.start_worker()
            self._pipeline_init = True
        else:
            self._gexe_.flush_pipeline()
        self._num_outstanding_cmd = 0
        self._cold_run = True
        self._gexe_.initialize_iter_pipeline(self._iter_repeat_count)
        num_cmd_to_push = self._queue_depth
        for _ in range(num_cmd_to_push):
            try:
                self._gexe_.execute_iter_pipeline()
                self._gexe_.execute_pipeline()
                self._num_outstanding_cmd = self._num_outstanding_cmd + 1
            except StopIteration:
                break

    def del_iter(self):
        """
        Method to close media iteration and release device.

        """
        t = tracer("del_iter")
        if self._pipeline_init == True:
            printf("{} iter del".format(self._pipename))
            self._gexe_.flush_pipeline()
            self._gexe_.stop_worker()
            self._gexe_.free_memory()
            self._gexe_.release_device()
            del self._gexe_
            self._pipeline_init = False

    def __run__(self):
        """
        Method to run mediapipe iterator over one batch of dataset.

        :returns : one batch of media graph processed output.
        :raises StopIteration: when complete dataset iteration and number repeat counnt is done.
        """
        t = tracer("mediapipe_run")
        if (self._num_outstanding_cmd > 0):
            # print(">> get_output")
            # start_time = time.perf_counter()
            outputs = self._gexe_.get_output()
            # end_time = time.perf_counter()
            # print("<< get_output : {:.6f}".format(end_time - start_time))
            self._num_outstanding_cmd = self._num_outstanding_cmd - 1
        else:
            raise StopIteration
        try:
            # print(">> iter")
            # start_time = time.perf_counter()
            self._gexe_.execute_iter_pipeline()
            # end_time = time.perf_counter()
            # print("<< iter time : {:.6f}".format(end_time - start_time))
            # print(">> exec pipe")
            # start_time = time.perf_counter()
            self._gexe_.execute_pipeline()
            # end_time = time.perf_counter()
            # print("<< exec pipe time : {:.6f}".format(end_time - start_time))
            self._num_outstanding_cmd = self._num_outstanding_cmd + 1
        except StopIteration:
            pass
        return outputs

    def __run_cpu__(self):
        t = tracer("mediapipe_run_cpu")
        return self._gexe_.get_output()

    def run(self):
        """
        Method to run mediapipe iterator over one batch of dataset.

        :returns : one batch of media graph processed output.
        :raises StopIteration: when complete dataset iteration and number repeat counnt is done.
        """
        return self.__execute()

    def as_iterator(self):
        """
        Method to get mediapipe iteratable object.

        :returns : mediapipe iteratable object.
        """
        if self._fw_type == mppy.fwType.SYNAPSE_FW:
            return iter(mediapipe=self)
        else:
            print("Pipe not iterable, use iterator")
            return None

    def get_batch_count(self):
        """
        Getter method to get media pipe number of batches in dataset.

        :returns : mediapipe number of batches presnet in dataset.
        """
        return self._gp.get_num_batches()
