from habana_frameworks.mediapipe.backend.nodes import opnode_tensor_info
from habana_frameworks.mediapipe.operators.media_nodes import MediaReaderNode
from habana_frameworks.mediapipe.media_types import dtype as dt
from habana_frameworks.mediapipe.media_types import lastBatchStrategy as lbs
from habana_frameworks.mediapipe.media_types import readerOutType as ro
from habana_frameworks.mediapipe.backend.utils import array_from_ptr
from habana_frameworks.mediapipe.backend.logger import printf
from habana_frameworks.mediapipe.media_types import mediaDeviceType as mdt
from habana_frameworks.mediapipe.operators.reader_nodes.reader_utils import dataset_shuffler
import numpy as np
import os
import glob
import time
import copy
import media_numpy_reader as mnr


def gen_npy_list(dir, pattern):
    return np.array(sorted(glob.glob(dir + "/{}".format(pattern))))


def get_max_file(file_list):
    return max(file_list, key=lambda x: os.stat(x).st_size)


broadcastable_params = [
    'dir', 'max_file', 'pattern',
]


class read_numpy_from_dir(MediaReaderNode):
    """
    Class defining numpy reader node.

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
        super().__init__(name, guid, device, inputs,
                         params, cparams, node_attr, fw_params)
        if (fw_params.device != mdt.LEGACY):
            if params['last_batch_strategy'] == lbs.NONE:
                if params['drop_remainder'] == True:
                    params['last_batch_strategy'] = lbs.DROP
                else:
                    if params['pad_remainder'] == True:
                        params['last_batch_strategy'] = lbs.PAD
                    else:
                        params['last_batch_strategy'] = lbs.CYCLIC
            del params['drop_remainder']
            del params['pad_remainder']

            if params['shard_once'] == None:
                if params['shuffle_across_dataset'] == True:
                    params['shard_once'] = False
                else:
                    params['shard_once'] = True
            del params['shuffle_across_dataset']
            return
        else:
            if params['last_batch_strategy'] != lbs.NONE or params['shard_once'] != None:
                raise ValueError(
                    "last_batch_strategy and shard_once not supported in legacy hpu pipe")

        self.batch_size = 1
        self.__params = copy.deepcopy(params)
        self.seed = self.__params['seed']
        del self.__params['seed']
        self.drop_remainder = self.__params['drop_remainder']
        del self.__params['drop_remainder']
        self.pad_remainder = self.__params['pad_remainder']
        del self.__params['pad_remainder']
        self.num_slices = self.__params['num_slices']
        del self.__params['num_slices']
        self.slice_index = self.__params['slice_index']
        del self.__params['slice_index']
        self.shuffle = self.__params['shuffle']
        del self.__params['shuffle']
        self.shuffle_across_dataset = self.__params['shuffle_across_dataset']
        del self.__params['shuffle_across_dataset']
        self.num_readers = self.__params["num_readers"]
        del self.__params['num_readers']
        self.is_modulo_slice = self.__params['is_modulo_slice']
        del self.__params['is_modulo_slice']
        if (self.num_readers < 1):
            raise ValueError("minimun one reader needed")
        if (self.num_readers > 8):
            raise ValueError("Num readers capped to 8")
        if (self.num_slices < 1):
            raise ValueError("num slice cannot be less then 1")
        if (self.slice_index >= self.num_slices):
            raise ValueError("slice_index cannot be >= num_slices")
        print("num_slices {} slice_index {}".format(
            self.num_slices, self.slice_index))
        print("random seed used ", self.seed)
        self.num_outputs = len(node_attr)
        self.dtype = []
        if (self.num_outputs > 2):
            raise ValueError("max outputs supported is two")
        for i in range(self.num_outputs):
            self.dtype.append(node_attr[i]['outputType'])
        if (self.num_outputs > 1):
            if (self.__params['file_list'] != [] and len(self.__params['file_list']) != self.num_outputs):
                raise ValueError(
                    "File list length should be equal to ouput operands expected but got ", len(self.__params['file_list']))
        else:
            if (self.__params['file_list'] != []):
                self.__params['file_list'] = [self.__params['file_list']]
        if (self.__params['file_list'] == []):
            self.__params['file_list'] = [[], []]

        bcst_params = self.broadcast_params(broadcastable_params,
                                            self.__params,
                                            self.num_outputs)
        self.readers = []
        for i in range(self.num_outputs):
            p = copy.deepcopy(self.__params)
            for key in bcst_params:
                p[key] = bcst_params[key][i]
            p['file_list'] = self.__params['file_list'][i]
            self.readers.append(_numpy_unit_reader_(p,
                                                    fw_params,
                                                    self.dtype[i],
                                                    self.num_readers))
        if (self.seed == -1):
            # max supported seed value is 32bit so modulo
            self.seed = int(time.time_ns() % (2**31 - 1))
        num_dataset = []
        for r in self.readers:
            num_dataset.append(r.get_unique_npys_count())
        if (np.max(num_dataset) != np.min(num_dataset)):
            raise ValueError("Readers length of dataset not matching")
        self.num_unique_ele = num_dataset[0]

        self.batch_size = fw_params.batch_size
        self.shuffler = dataset_shuffler(self.seed,
                                         self.shuffle,
                                         self.slice_index,
                                         self.num_slices,
                                         self.batch_size,
                                         self.num_unique_ele,
                                         self.drop_remainder,
                                         self.pad_remainder,
                                         self.shuffle_across_dataset,
                                         self.is_modulo_slice)

    def broadcast_params(self, key_list, dictionary, broadcast_lenght):
        bcst_dict = {}
        for key in key_list:
            if (not isinstance(dictionary[key], list)):
                dictionary[key] = [dictionary[key]]
            if len(dictionary[key]) != broadcast_lenght:
                if (len(dictionary[key]) == 1):
                    for i in range(broadcast_lenght - 1):
                        dictionary[key].append(dictionary[key][0])
            bcst_dict[key] = dictionary[key]
        return bcst_dict

    def __del__(self):
        if (self.device == "cpu"):
            return
        for r in self.readers:
            del r

    def gen_output_info(self):
        """
        Method to generate output type information.

        :returns : output tensor information of type "opnode_tensor_info".
        """
        out_info = []
        for r in self.readers:
            out_info.append(r.gen_output_info()[0])
        return out_info

    def get_largest_file(self):
        """
        Method to get largest media in the dataset.

        returns: largest media element in the dataset.
        """
        fileList = []
        for r in self.readers:
            fileList.append(r.fileList())
        self.max_file = get_max_file(fileList)
        return self.max_file

    def get_media_output_type(self):
        return ro.BUFFER_LIST

    def __len__(self):
        """
        Method to get dataset length.

        returns: length of dataset in units of batch_size.
        """
        elements = self.shuffler.get_num_iterable_elements()
        if elements % self.batch_size:
            raise ValueError(
                "iterable elements should be multiple of batch size")
        return elements // self.batch_size

    def __iter__(self):
        """
        Method to initialize iterator.

        """
        idxs = self.shuffler.gen_idx_list()
        for r in self.readers:
            r.iter(idxs)
        return self

    def __next__(self):
        """
        Method to get one batch of dataset ouput from iterator.

        """
        out_bufs = []
        stopException = 0
        for r in self.readers:
            try:
                out_bufs.append(r.next())
            except StopIteration:
                stopException += 1
        if (stopException == self.num_outputs):
            raise StopIteration
        elif (len(out_bufs) == self.num_outputs):
            # print(out_bufs)
            return out_bufs
        else:
            raise RuntimeError("readers not synchronized")


class _numpy_unit_reader_():
    """
    Class defining numpy reader node.

    """

    def __init__(self, params, fw_params, dtype, num_readers):
        """
        Constructor method.

        :params name: node name.
        :params guid: guid of node.
        :params guid: device on which this node should execute.
        :params params: node specific params.
        :params cparams: backend params.
        :params out_info: node output information
        """
        self.batch_size = 1
        self.__params = copy.deepcopy(params)
        self.file_list = self.__params['file_list']
        self.dir = self.__params['dir']
        self.pattern = self.__params['pattern']

        self.max_file = self.__params['max_file']
        self.dense = self.__params['dense']
        self.num_readers = num_readers
        self.output_dtype = dtype
        if (self.file_list == [] and self.dir != ""):
            print("Finding images ...", end=" ")
            self.npy_unique_list = gen_npy_list(self.dir, self.pattern)
            print("Done!")
        elif (self.file_list != [] and self.dir == ""):
            self.npy_unique_list = np.array(self.file_list)
        elif (self.file_list == [] and self.dir == ""):
            raise ValueError("Atleast file_list or dir must be shared")
        else:
            raise ValueError("Only one file_list or dir must be shared")
        self.num_unique_ele = len(self.npy_unique_list)
        if (self.num_unique_ele == 0):
            raise ValueError("npys list empty !!!")
        print("Total npy files {} ".format(self.num_unique_ele))
        # print(self.num_slices)
        if (self.max_file == ""):
            print("Finding largest file ...")
            self.max_file = get_max_file(self.npy_unique_list)
        print("largest file is ", self.max_file)
        self.num_outstanding_cmds = 0
        self.reader = None

        self.batch_size = fw_params.batch_size
        self.queue_depth = fw_params.queue_depth
        if (self.dense == True):
            self.shape = np.load(self.max_file).shape
            if (not isinstance(self.shape, list)):
                self.shape = list(self.shape)
            self.shape = self.shape[::-1]
            self.shape.append(self.batch_size)
        else:
            self.shape = [0, 0, 0, self.batch_size]
        # self.num_readers = 1
        self.reader = mnr.NumpyReader(self.queue_depth, self.num_readers, self.batch_size,
                                      os.stat(self.max_file).st_size)
        self.reader.StartWorker()

    def __del__(self):
        if (self.reader != None):
            self.reader.StopWorker()
            del self.reader

    def gen_output_info(self):
        """
        Method to generate output type information.

        :returns : output tensor information of type "opnode_tensor_info".
        """
        out_info = []
        o = opnode_tensor_info(self.output_dtype,
                               np.array(self.shape, dtype=np.uint32),
                               "")
        out_info.append(o)
        return out_info

    def get_largest_file(self):
        """
        Method to get largest media in the dataset.

        returns: largest media element in the dataset.
        """
        return self.max_file

    def get_media_output_type(self):
        return ro.BUFFER_LIST

    def get_unique_npys_count(self):
        return self.num_unique_ele

    def iter(self, list_shuffled_sliced_idxs):
        """
        Method to initialize iterator.

        """
        self.reader.flush()
        self.num_outstanding_cmds = 0
        self.npy_iter_list = self.npy_unique_list[list_shuffled_sliced_idxs]
        self.num_iter_list = len(self.npy_iter_list)

        self.iter_loc = 0
        while (self.num_outstanding_cmds < self.queue_depth):
            if self.iter_loc > (self.num_iter_list - 1):
                break
            else:
                start = self.iter_loc
                end = self.iter_loc + self.batch_size
                npy_list = self.npy_iter_list[start:end]
                self.iter_loc = self.iter_loc + self.batch_size
                self.reader.SubmitFiles(npy_list)
                self.num_outstanding_cmds = self.num_outstanding_cmds + 1
        return self

    def next(self):
        """
        Method to get one batch of dataset ouput from iterator.

        """
        if self.num_outstanding_cmds == 0:
            raise StopIteration
        output = self.reader.WaitForCompletion()
        self.num_outstanding_cmds = self.num_outstanding_cmds - 1
        if self.iter_loc > (self.num_iter_list - 1):
            pass
        else:
            start = self.iter_loc
            end = self.iter_loc + self.batch_size
            npy_list = self.npy_iter_list[start:end]
            self.iter_loc = self.iter_loc + self.batch_size
            self.reader.SubmitFiles(npy_list)
            self.num_outstanding_cmds = self.num_outstanding_cmds + 1
        out_bufs = []
        for o in output:
            out_bufs.append(array_from_ptr(o.pBuffer,
                                           o.typeStr,
                                           tuple(o.shape[:o.numDims])))
        for o in out_bufs:
            if (o.dtype != self.output_dtype):
                raise ValueError("Datatype mismatch file contains dtype{} reader expected dtype {}".format(
                    o.dtype, self.output_dtype))
        if (self.dense):
            output = np.stack(out_bufs)
        else:
            output = out_bufs
        return output
