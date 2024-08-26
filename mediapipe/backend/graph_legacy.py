# this file is responsible for graph handling of mediapipe
from habana_frameworks.mediapipe.backend.tracing import media_tracer, tracer
from habana_frameworks.mediapipe.backend.cal import media_manager, pipe_manager
from habana_frameworks.mediapipe.backend.nodes import construct_bidirectional_graph
from habana_frameworks.mediapipe.backend.nodes import pipe_fw_params
from habana_frameworks.mediapipe.operators.media_nodes import MediaConstantNode, MediaDummyNode
from habana_frameworks.mediapipe.operators.media_nodes import MediaReaderNode
from habana_frameworks.mediapipe.operators.media_nodes import MediaFuncDataNode
from habana_frameworks.mediapipe.operators.media_nodes import MediaDecoderNode
from habana_frameworks.mediapipe.operators.media_nodes import MediaHPUNode
from habana_frameworks.mediapipe.operators.media_nodes import MediaCPUNode
from habana_frameworks.mediapipe.operators.media_nodes import CPPNode
from habana_frameworks.mediapipe.operators.decoder_nodes.decoder_nodes import image_decoder
from habana_frameworks.mediapipe.operators.decoder_nodes.decoder_nodes import _video_decoder
import time


class graph_signature(object):
    """
    Class defining hpu graph nodes

    """

    def __init__(self, input_tensors, const_tensors, ops, output_tensors):
        """
        Constructor method.

        """
        self.input_tensors = input_tensors
        self.const_tensors = const_tensors
        self.output_tensors = output_tensors
        self.input_unique_tensors = self.find_unique_tensor_list(input_tensors)
        self.const_unique_tensors = self.find_unique_tensor_list(const_tensors)
        self.output_unique_tensors = self.find_unique_tensor_list(
            output_tensors)
        self.ops = ops

    def reorder_io_unique_tensors(self, input_unique_tensors, output_unique_tensors):
        """
        Setter method to set media graph input tensor.

        """
        if (len(self.input_unique_tensors) != len(
                input_unique_tensors)):
            raise RuntimeError("Mismatch in hpu input tensor")
        if (len(self.output_unique_tensors) != len(
                output_unique_tensors)):
            raise RuntimeError("Mismatch in hpu output tensor")
        self.input_unique_tensors = self.find_unique_tensor_list(
            input_unique_tensors)
        self.output_unique_tensors = self.find_unique_tensor_list(
            output_unique_tensors)

    def find_unique_tensor_list(self, tensors):
        tensor_unique = []
        for t in tensors:
            if len(tensor_unique) == 0:
                tensor_unique.append(t)
            else:
                match = 0
                for tu in tensor_unique:
                    if (tu.name == t.name):
                        match = match + 1
                if match == 0:
                    tensor_unique.append(t)
        return tensor_unique


class graph_processor(object):
    """
    Class defining compile time processing of media nodes.

    """

    def __init__(self, device_type, output_tensors, fw_type, proxy):
        """
        Constructor method.

        """
        self._device_type = device_type
        self._output_tensors = output_tensors
        self._fw_type = fw_type
        self._mm_ = media_manager(self._device_type)
        self._ops = []
        self._readers_ = []
        self._const_inputs_ = []
        self._func_inputs_ = []
        self._decoders_ = []
        self._hpu_ops_ = []
        self._cpu_ops_ = []
        self._dummy_ops_ = []
        self._cpp_nodes_ = []
        self._ngops_output_tensors_ = None
        self._is_processed_ = False
        self._is_segmented_ = False
        self._hpu_graph_ = None
        self._hpu_tensor_info_ = None
        self._hpu_to_py_output_map_ = None

        tensors = self._output_tensors.copy()
        for t in tensors:
            t.dst_op.append(None)
        self._ops = construct_bidirectional_graph(tensors, [])

    def create_opnodes(self, batch_size, queue_depth, num_threads):
        fw_params = pipe_fw_params(
            self._device_type, batch_size, queue_depth, num_threads)
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
        # update pipe specific params and generate output information
        ops = self._ops
        for op in ops:
            op.generate_node_info()
        # since we suppport only one decoder and reader check it here
        media_decoders = self._decoders_
        media_readers = self._readers_
        if (len(media_decoders) > 1):
            raise ValueError("Single decoder supported")
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
                self._decoders_.append(o)
            elif isinstance(o, MediaHPUNode):
                self._hpu_ops_.append(o)
            elif isinstance(o, MediaCPUNode):
                self._cpu_ops_.append(o)
                if isinstance(o, CPPNode):
                    self._cpp_nodes_.append(o)
            elif isinstance(o, MediaDummyNode):
                self._dummy_ops_.append(o)
            else:
                raise RuntimeError("invalid operator")
        self._media_graph_ = graph_signature(self._func_inputs_,
                                             self._const_inputs_,
                                             self._ops,
                                             self._output_tensors)

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
        self._ngops_output_tensors_ = []
        for o in self._output_tensors:
            if (not (isinstance(o.src_op, MediaHPUNode) or isinstance(o.src_op, MediaDecoderNode))):
                self._ngops_output_tensors_.append(o)

        self._is_segmented_ = True

    def compile(self):
        """
        Method to compile graph.

        """
        hpu_graph = self.get_hpu_graph()
        decoders = self.get_decoder_ops()
        decoder_op = None
        if (len(decoders) == 1):
            decoder_op = decoders[0]
        # INFO: compile recipe only if decoder or hpu nodes present
        if (len(hpu_graph.ops) > 0 or len(decoders) > 0):
            self._mm_.compile(decoder_op, hpu_graph)
        else:
            raise RuntimeError("Minmum 1 decoder or hpu node needed")
        return self._mm_.get_recipe()

    def get_cpp_nodes(self):
        """
        Getter method to get metadata processors.
        """
        return self._cpp_nodes_

    def get_recipe(self):
        """
        Getter method to get graph recipe.

        """
        return self._mm_.get_recipe()

    def get_media_graph(self):
        """
        Getter method to get full media graph.

        """
        return self._media_graph_

    def get_hpu_graph(self):
        """
        Getter method to get media graph to be run on hpu.

        """
        if (self._hpu_graph_ != None):
            return self._hpu_graph_
        input_tensors = []
        const_tensors = []
        output_tensors = []
        if (len(self._hpu_ops_) == 0):
            if (len(self._decoders_) != 0):
                # INFO: no hpu ops present so register dec ouput as in/out
                decoder_op = self.get_decoder_ops()[0]
                input_tensors.append(decoder_op.output_tensors[0])
                output_tensors.append(decoder_op.output_tensors[0])
        else:
            for ho in self._hpu_ops_:
                # INFO: since we have assumned hpu graph to be last part of graph we can safely take this check to get outputs
                for o in ho.output_tensors:
                    for d in o.dst_op:
                        if (d == None):
                            output_tensors.append(o)
                for i in ho.input_tensors:
                    if (not isinstance(i.src_op, MediaHPUNode)):
                        if (isinstance(i.src_op, MediaConstantNode)):
                            const_tensors.append(i)
                        else:
                            input_tensors.append(i)
            # handle decoder output taking as graph output
            for d in self._decoders_:
                for ot in d.output_tensors:
                    if ot in self._output_tensors:
                        output_tensors.append(ot)

        self._hpu_graph_ = graph_signature(input_tensors,
                                           const_tensors,
                                           self._hpu_ops_,
                                           output_tensors)
        return self._hpu_graph_

    def process_recipe(self):
        if (self._hpu_tensor_info_ != None):
            return self._hpu_tensor_info_
        hpu_graph = self.get_hpu_graph()
        num_inputs = len(hpu_graph.input_unique_tensors)
        num_outputs = len(hpu_graph.output_unique_tensors)
        input_tensors, output_tensors = self._mm_.get_hpu_tensor_info(self.get_recipe(),
                                                                      num_outputs,
                                                                      num_inputs)
        if (len(input_tensors) != num_inputs):
            raise ValueError("hpu input tensor count mismatch")
        if (len(output_tensors) != num_outputs):
            raise ValueError("hpu output tensor count mismatch")
        in_tensors_org = []
        out_tensors_org = []
        input_names = []
        for it in input_tensors:
            for hit in hpu_graph.input_unique_tensors:
                if (it.name == hit.name):
                    in_tensors_org.append(hit)
                    input_names.append(it.name)
                    break
        if (len(in_tensors_org) != num_inputs):
            raise RuntimeError("couldnot map all hpu input tensors")
        for ot in output_tensors:
            for hot in hpu_graph.output_unique_tensors:
                # this is needed because backend appends srcport to name
                len_srcport = len(ot.name.split("_")[-1]) + 1
                if ot.name[:-len_srcport] in input_names:
                    hot_name = hot.name+"_"+str(hot.src_port)
                else:
                    hot_name = hot.src_op.name+"_"+str(hot.src_port)
                if (ot.name == hot_name):
                    hot.shape = ot.shape
                    hot.np_shape = ot.np_shape
                    out_tensors_org.append(hot)
                    break
        hpu_graph.reorder_io_unique_tensors(in_tensors_org, out_tensors_org)
        hpu_graph = self.get_hpu_graph()
        self._hpu_to_py_output_map_ = []
        for o in self._output_tensors:
            mapped = False
            for i in range(len(hpu_graph.output_unique_tensors)):
                if (o.name == hpu_graph.output_unique_tensors[i].name):
                    self._hpu_to_py_output_map_.append(i)
                    mapped = True
                    break
            if (mapped):
                continue
            for i in range(len(self._ngops_output_tensors_)):
                if (o.name == self._ngops_output_tensors_[i].name):
                    self._hpu_to_py_output_map_.append(
                        i + len(hpu_graph.output_tensors))
                    mapped = True
                    break
            if (not mapped):
                raise RuntimeError("unable to map {} in graph", o.name)

    def get_ops(self):
        """
        Getter method to get list of media ops present in graph.

        """
        return self._ops

    def get_output_tensors(self):
        """
        Getter method to get list of media output tensors.

        """
        return self._output_tensors

    def get_reader_ops(self):
        """
        Getter method to get list of media reader nodes.

        """
        return self._readers_

    def get_num_batches(self):
        """
        Getter method to get list of media reader nodes.

        """
        reader = self.get_reader_ops()[0]
        return len(reader)

    def get_const_ops(self):
        """
        Getter method to get list of constant nodes.

        """
        return self._const_inputs_

    def get_func_ops(self):
        """
        Getter method to get list of function nodes.

        """
        return self._func_inputs_

    def get_decoder_ops(self):
        """
        Getter method to get list of decoder nodes.

        """
        return self._decoders_

    def get_hpu_ops(self):
        """
        Getter method to get list of hpu nodes.

        """
        return self._hpu_ops_

    def get_cpu_ops(self):
        """
        Getter method to get list of cpu nodes.

        """
        return self._cpu_ops_

    def get_hpu_to_py_output_map(self):
        """
        Getter method to get list of non pu graph output tensors.

        """
        return self._hpu_to_py_output_map_

    def get_ngop_output_tensors(self):
        """
        Getter method to get list of non pu graph output tensors.

        """
        return self._ngops_output_tensors_


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
        self._hpu_graph_ = graph_processor.get_hpu_graph()
        self._npop_output_tensors_ = graph_processor.get_ngop_output_tensors()
        self._pm_ = pipe_manager(queue_depth,
                                 batch_size,
                                 fw_type,
                                 proxy,
                                 python_proxy,
                                 self._npop_output_tensors_,
                                 graph_processor.get_output_tensors(),
                                 graph_processor.get_hpu_to_py_output_map())
        self._graph_processor_ = graph_processor
        self._exec_ = None
        self._const_exec_ = None
        self._decoder_op_ = None
        self.recipe = self._graph_processor_.get_recipe()
        decoder_op = self._graph_processor_.get_decoder_ops()
        if (len(decoder_op) == 1):
            self._decoder_op_ = self._graph_processor_.get_decoder_ops()[0]
        self._all_output_tensors_ = self._get_all_output_tensors_()
        self._iter_ops_ = self._graph_processor_.get_reader_ops()
        self._iters_ = []
        self.tracer = media_tracer()

        # Register all cpp nodes to cpp backend
        for p in graph_processor.get_cpp_nodes():
            p.add_to_pipeline(self._pm_)

    def start_worker(self):
        """
        Method to start backend worker.

        """
        self._pm_.start_worker()

    def stop_worker(self):
        """
        Method to stop backend worker.

        """
        self._pm_.stop_worker()

    def acquire_device(self, device):
        """
        Method to acquire device.

        """
        self._pm_.acquire_device(device)

    def release_device(self, ):
        """
        Method to release device.

        """
        self._pm_.release_device()

    def initialize_memory(self):
        """
        Method to initialize all backend memory.

        """
        if (self._decoder_op_ != None):
            node = self._decoder_op_
            while (not isinstance(node.input_tensors[0].src_op, MediaReaderNode)):
                err_msg = "Error retreving reader node for decoder "
                node = node.input_tensors[0].src_op
                if (len(node.input_tensors) > 1):
                    raise ValueError(err_msg +
                                     "multi input node")
                if (len(node.output_tensors) > 1):
                    raise ValueError(err_msg +
                                     "found multi output node")
            self._pm_.decoder_init(
                self._decoder_op_, node.input_tensors[0].src_op)

        if (self.recipe != None):
            self._pm_.media_init(self._graph_processor_.get_recipe(),
                                 len(self._hpu_graph_.output_tensors),
                                 len(self._hpu_graph_.input_tensors))
        # generate the pipeline's for constants and cpu
        self._const_exec_ = self._generate_const_pipeline_()
        self._exec_ = self._generate_pipeline_()

        self._pm_.initialize_host_buffer()

    def free_memory(self):
        """
        Method to free all backend memory.

        """
        self._pm_.free_host_buffer()
        if (self._decoder_op_ != None):
            self._pm_.decoder_deinit()
        if (self.recipe != None):
            self._pm_.media_deinit()

    def flush_pipeline(self):
        """
        Method to flush pending command in pipe.

        """
        self._pm_.flush_pipeline()
        for t in self._all_output_tensors_:
            t.clear_data()

    def _generate_const_pipeline_(self):
        """
        Method to generate constant node pipeline.

        """
        const_exec = []
        for i in self._hpu_graph_.input_tensors:
            if (isinstance(i.src_op, MediaConstantNode)):
                const_exec.append(i.src_op)
        return const_exec

    def _generate_pipeline_(self):
        """
        Method to generate E2E pipeline.

        """
        exec = []
        ops = self._graph_processor_.get_ops()
        for op in ops:
            if (isinstance(op, MediaFuncDataNode)
                    or isinstance(op, MediaCPUNode)
                    or isinstance(op, MediaDummyNode)):
                exec.append(op)
            elif (isinstance(op, MediaConstantNode)):
                for o in op.output_tensors:
                    for d in o.dst_op:
                        if (isinstance(d, MediaFuncDataNode)
                                or isinstance(d, MediaCPUNode)):
                            exec.append(op)

        # INFO: last node is always hpu node running in c as of today
        c_hpu = c_hpu_node("c_hpu0",
                           self._decoder_op_,
                           self._hpu_graph_.input_unique_tensors,
                           self._hpu_graph_.input_tensors,
                           self._npop_output_tensors_,
                           self._pm_.run_hpu)
        exec.append(c_hpu)
        return exec

    def __generate_cpu_pipeline_(self):
        """
        Method to generate cpu alone pipeline.

        """
        cpu_exec = []
        cpu_ops = self._graph_processor_.get_cpu_ops()
        for op in cpu_ops:
            cpu_exec.append(op)
        return cpu_exec

    def _get_all_output_tensors_(self):
        """
        Method to get output tensors of the pipeline.

        """
        ops = self._graph_processor_.get_ops()
        tensors = []
        for op in ops:
            for o in op.output_tensors:
                tensors.append(o)
        return tensors

    # below are the executors to vall to get execution of nodes
    def initialize_iter_pipeline(self, repeat_count):
        """
        Method to initialize iterator of the pipe.

        """
        self._pm_.init_iterator()
        self._iters_ = []
        for ip in self._iter_ops_:
            self._iters_.append(iter(ip))
        self._iter_repeat_idx_ = 0
        self._iter_repeat_count_ = repeat_count

    def execute_iter_pipeline(self):
        """
        Method to execute iterator.

        """
        t = tracer("execute_iter_pipeline")
        for i in range(len(self._iters_)):
            try:
                outputs = next(self._iters_[i])
            except StopIteration:
                # because of queue depth we need to keep breaking here
                self._iter_repeat_idx_ = self._iter_repeat_idx_ + 1
                if (self._iter_repeat_count_ != -1 and self._iter_repeat_idx_ >= self._iter_repeat_count_):
                    raise StopIteration
                self._iter_ = iter(self._iter_ops_[i])
                outputs = next(self._iters_[i])
            self._iter_ops_[i].push_output_buffers(outputs)

    def execute_const_pipeline(self):
        """
        Method to execute constant pipeline.

        """
        for e in self._const_exec_:
            outputs = e()
            e.push_output_buffers(outputs)

    def execute_pipeline(self):
        """
        Method to execute E2E pipeline.

        """
        t = tracer("execute_pipeline")
        for e in self._exec_:
            if (isinstance(e, MediaFuncDataNode)):
                name = e.name
                for i in range(len(e.output_tensors[0].dst_op)):
                    try:
                        n = e.name+e.output_tensors[0].dst_op[i].name
                        name = n
                        break
                    except:
                        pass
            else:
                name = e.name
            self.tracer.start_trace(name)
            # print(">>> exec run", e.name)
            # start_time = time.perf_counter()
            inputs = e.fetch_input_buffers()
            outputs = e(*inputs)
            e.push_output_buffers(outputs)
            self.tracer.end_trace(name)
            # end_time = time.perf_counter()
            # print("<<< exec run {} time {}".format(e.name, end_time-start_time))

    def get_output(self):
        t = tracer("get_output")
        return self._pm_.get_output()


# raw hpu node: non standard special op to handle c hpu runs
class c_hpu_node(MediaCPUNode):
    def __init__(self, name, decoder_op, unique_inputs, all_inputs, ngop_output_tensors, run_hpu):
        super().__init__(
            name, "c_hpu0", "", [], {}, {}, None, None)
        self.populate_output_tensors([])
        self.run_hpu = run_hpu
        self.decoder_op = decoder_op
        self.get_ngops_buf_funcs = []
        for t in ngop_output_tensors:
            self.get_ngops_buf_funcs.append(t.data_read)

        self.var_buf_funcs = []
        for i in unique_inputs:
            if (not isinstance(i.src_op, MediaDecoderNode)):
                self.var_buf_funcs.append(i.data_read)

        repeat_inputs = self.get_repeated_inputs(unique_inputs, all_inputs)
        self.rpt_buf_funcs = []
        for i in repeat_inputs:
            if (not isinstance(i.src_op, MediaDecoderNode)):
                self.rpt_buf_funcs.append(i.data_read)

    def get_repeated_inputs(self, unique_inputs, all_inputs):
        ui = unique_inputs.copy()
        repeat_inputs = []
        for a in all_inputs:
            idx = -1
            for i in range(len(ui)):
                if (ui[i].name == a.name):
                    idx = i
                    break
            if (idx >= 0):
                ui.pop(idx)
            else:
                repeat_inputs.append(a)
        return repeat_inputs

    def gen_output_info(self):
        pass

    def fetch_input_buffers(self):
        file_list = None
        rand_crp = None
        vid_offset = None
        crop_after_resize = False
        resample_idx = None
        is_gather_nd_for_decode = False

        if (self.decoder_op != None):
            file_list = self.decoder_op.input_tensors[0].data_read()
            # start_time1 = time.perf_counter()
            if isinstance(self.decoder_op, image_decoder):
                is_gather_nd_for_decode = False
                if (len(self.decoder_op.input_tensors) > 1):
                    rand_crp = self.decoder_op.input_tensors[1].data_read()

            elif isinstance(self.decoder_op, _video_decoder):

                is_gather_nd_for_decode = self.decoder_op.get_dec_params()[
                    "is_gather_nd"]
                vid_offset = self.decoder_op.input_tensors[1].data_read()
                in_len = len(self.decoder_op.input_tensors)
                if (in_len > 2):
                    resample_idx = self.decoder_op.input_tensors[2].data_read()
                if (in_len > 3):
                    rand_crp = self.decoder_op.input_tensors[3].data_read()

            else:
                raise RuntimeError("unknown decoder ", self.decoder_op)

            crop_after_resize = self.decoder_op.get_dec_params()[
                "is_crop_after_resize"]

        ngop_np_buf = []
        for f in self.get_ngops_buf_funcs:
            ngop_np_buf.append(f())

        var_np_buf = []
        for f in self.var_buf_funcs:
            var_np_buf.append(f())
        for f in self.rpt_buf_funcs:
            f()
        return file_list, rand_crp, crop_after_resize, vid_offset, resample_idx, var_np_buf, ngop_np_buf, is_gather_nd_for_decode

    def push_output_buffers(self, outputs):
        pass

    def __call__(self, file_list, rand_crp, is_crp_after_resize, vid_offset, resample_idx, var_np_buf, ngop_np_buf, is_gather):
        self.run_hpu(file_list, rand_crp, is_crp_after_resize,
                     vid_offset, resample_idx, var_np_buf, ngop_np_buf, is_gather)
        return []
