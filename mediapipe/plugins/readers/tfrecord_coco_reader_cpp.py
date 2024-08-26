import sys
import os
import random
from habana_frameworks.mediapipe.backend.utils import get_str_dtype, get_numpy_dtype, get_media_dtype  # NOQA
from habana_frameworks.mediapipe.operators.reader_nodes.reader_nodes import media_ext_reader_op_impl  # NOQA
from habana_frameworks.mediapipe.operators.reader_nodes.reader_nodes import media_ext_reader_op_tensor_info  # NOQA
from habana_frameworks.mediapipe.media_types import readerOutType as ro  # NOQA
from habana_frameworks.mediapipe.media_types import dtype as dt  # NOQA
import numpy as np
import media_tfrecord_read as mtfr


tfr_reader_params = {
    'dir': "/",
    'enable_shuffle': True,
    'batch_size': 256,
    'label_dtype': dt.FLOAT32,
    'num_slices': 1,
    'slice_index': 0,
    'fieldPattern': {"height": "image/height", "width": "image/width", "sourceId": "image/source_id", "imageEncoded": "image/encoded", "filename": "image/filename", "caption": "image/caption", "format": "image/format", "keySha256": "image/key/sha256", "objectClassLabel": "image/object/class/label", "objectBboxYmin": "image/object/bbox/ymin", "objectBboxYmax": "image/object/bbox/ymax", "objectBboxXmin": "image/object/bbox/xmin", "objectBboxXmax": "image/object/bbox/xmax", "objectIsCrowd": "image/object/is_crowd", "objectArea": "image/object/area", "objectMask": "image/object/mask", "objectClassText": "image/object/class/text"}
}


class tfr_coco_reader_cpp(media_ext_reader_op_impl):
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
        self.mustFieldPattern = ['imageEncoded', 'objectClassLabel',
                                 'objectBboxYmin', 'objectBboxYmax', 'objectBboxXmin', 'objectBboxXmax']
        for mfp in self.mustFieldPattern:
            if mfp not in self.fieldPattern:
                print("Error: Required Field not provided :", mfp)
                exit(1)
        self.dataset = mtfr.TFRecordDatasetReader(
            self.data_dir, 2, self.fieldPattern, self.batch_size, self.num_slices, self.slice_index, self.enable_shuffle)

        self.batch_size = fw_params.batch_size

    def __iter__(self):
        self.dataset.dataSetInit()
        return self

    def __len__(self):
        # it should return num_batches -- which is unknown
        return 0

    def __next__(self):
        img_batch, bbox_labels_batch, bbox_xmin_batch, bbox_xmax_batch, bbox_ymin_batch, bbox_ymax_batch = self.dataset.getNextBatchCoco()
        bbox_labels_batch = np.array(bbox_labels_batch,  dtype=object)
        bbox_xmin_batch = np.array(bbox_xmin_batch,  dtype=object)
        bbox_xmax_batch = np.array(bbox_xmax_batch,  dtype=object)
        bbox_ymin_batch = np.array(bbox_ymin_batch,  dtype=object)
        bbox_ymax_batch = np.array(bbox_ymax_batch,  dtype=object)

        if(len(img_batch) < self.batch_size):
            raise StopIteration
        # return img_batch, bbox_labels_batch, bbox_xmin_batch,bbox_xmax_batch, bbox_ymin_batch, bbox_ymax_batch
        return img_batch, bbox_labels_batch

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

