import sys
import os
import random
from habana_frameworks.mediapipe.backend.utils import get_str_dtype, get_numpy_dtype, get_media_dtype  # NOQA
from habana_frameworks.mediapipe.operators.reader_nodes.reader_nodes import media_ext_reader_op_impl  # NOQA
from habana_frameworks.mediapipe.operators.reader_nodes.reader_nodes import media_ext_reader_op_tensor_info  # NOQA
from habana_frameworks.mediapipe.media_types import readerOutType as ro  # NOQA
from habana_frameworks.mediapipe.media_types import dtype as dt  # NOQA
import numpy as np
import media_tfrecord_read as mtfr  # NOQA


tfr_reader_params = {
    'dir': "/",
    'enable_shuffle': True,
    'batch_size': 256,
    'label_dtype': dt.FLOAT32,
    'num_slices': 1,
    'slice_index': 0,
    'fieldPattern': {"width": "image/width", "height": "image/height", "channels": "image/channels", "classLabel": "image/class/label",  "filename": "image/filename", "format": "image/format", "colorspace": "image/colorspace", "synset": "image/class/synset", "text": "image/class/text", "imageEncoded": "image/encoded"}
}


class tfr_reader_cpp(media_ext_reader_op_impl):
    def __init__(self, params):
        params = params['priv_params']
        self.data_dir = params['dir']
        self.enable_shuffle = params['enable_shuffle']
        self.batch_size = params['batch_size']
        self.metadata_dtype = params["label_dtype"]
        self.num_slices = params['num_slices']
        self.slice_index = params['slice_index']
        self.metadata_dtype_np = get_numpy_dtype(self.metadata_dtype)
        # print("tfr_reader_cpp:input_path:",self._input_path)
        self.fieldPattern = params['fieldPattern']
        self.mustFieldPattern = ['imageEncoded', 'classLabel']
        for mfp in self.mustFieldPattern:
            if mfp not in self.fieldPattern:
                print("Error: Required Field not provided :", mfp)
                exit(1)
        self.dataset = mtfr.TFRecordDatasetReader(
            self.data_dir, 1, self.fieldPattern, self.batch_size, self.num_slices, self.slice_index, self.enable_shuffle)

        self.batch_size = fw_params.batch_size

    def __iter__(self):
        self.dataset.dataSetInit()
        return self

    def __len__(self):
        # it should return num_batches -- which is unknown
        return 0

    def __next__(self):
        img_batch, label_batch = self.dataset.getNextBatchImnet()
        label_batch = np.array(label_batch)
        label_batch = np.reshape(label_batch, (-1, 1))
        label_batch = label_batch.astype(self.metadata_dtype_np)
        if(len(img_batch) < self.batch_size):
            raise StopIteration
        return img_batch, label_batch

    def get_media_output_type(self):
        return ro.ADDRESS_LIST

    def get_largest_file(self):
        return ""

    def gen_output_info(self):
        out_info = []
        o = media_ext_reader_op_tensor_info(
            dt.NDT, np.array([self.batch_size], dtype=np.uint32), "")
        out_info.append(o)
        o = media_ext_reader_op_tensor_info(
            self.metadata_dtype, np.array([self.batch_size], dtype=np.uint32), "")
        out_info.append(o)
        return out_info

