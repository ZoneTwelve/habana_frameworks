# Copyright 2018 Google. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
# Changes:
# - Ported to TF 2.2
# - made interable to media pipe
# - Added absolute paths in imports

from habana_frameworks.mediapipe.media_types import dtype as dt  # NOQA
from habana_frameworks.mediapipe.backend.utils import get_str_dtype, get_numpy_dtype, get_media_dtype  # NOQA
from habana_frameworks.mediapipe.operators.reader_nodes.reader_nodes import media_ext_reader_op_impl  # NOQA
from habana_frameworks.mediapipe.operators.reader_nodes.reader_nodes import media_ext_reader_op_tensor_info  # NOQA
from habana_frameworks.mediapipe.media_types import readerOutType as ro  # NOQA
import os
import glob
import numpy as np
import tensorflow as tf


_SHUFFLE_BUFFER = 1000
_NUM_TRAIN_FILES = 1024


def _get_crop_param(image_buffer, bbox, sample_distorted_bounding_box_seed=0):
    # A large fraction of image datasets contain a human-annotated bounding box
    # delineating the region of the image containing the object of interest.  We
    # choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.image.extract_jpeg_shape(image_buffer), bounding_boxes=bbox,
        seed=sample_distorted_bounding_box_seed, min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33], area_range=[0.05, 1.0],
        max_attempts=100, use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Reassemble the bounding box in the format the crop op requires.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

    return crop_window


def parse_record(raw_record):
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''), }
    sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update({k: sparse_float32 for k in [
        'image/object/bbox/xmin', 'image/object/bbox/ymin',
        'image/object/bbox/xmax', 'image/object/bbox/ymax']})

    features = tf.io.parse_single_example(
        serialized=raw_record, features=feature_map)
    image_buffer = features['image/encoded']
    # Tensor("ParseSingleExample/ParseExample/ParseExampleV2:14", shape=(), dtype=string)

    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(a=bbox, perm=[0, 2, 1])
    crop_window = _get_crop_param(image_buffer, bbox)

    # Subtract one so that labels are in [0, 1000), and cast to float32 for
    # Keras model.
    label = tf.cast(tf.cast(tf.reshape(label, shape=[1]), dtype=tf.int32) - 1,
                    dtype=tf.int32)

    # image_buffer = tf.cast(tf.reshape(image_buffer, shape=[1]), dtype=tf.uint8)
    # image_buffer = tf.reshape(image_buffer, shape=[1])
    return image_buffer, crop_window, label


tfr_reader_params = {
    'dir': "/",
    'enable_shuffle': True,
    'batch_size': 256,
    'label_dtype': dt.FLOAT32,
    'num_slices': 1,
    'slice_index': 0,
    'enable_cache': False,
    'enable_experimental_deterministic': False,
    'experimental_preloading': False
}


class tfr_reader(media_ext_reader_op_impl):
    def __init__(self, params, fw_params):
        params = params['priv_params']
        self.data_dir = params['dir']
        self.enable_cache = params['enable_cache']
        self.enable_experimental_deterministic = params['enable_experimental_deterministic']
        self.experimental_preloading = params['experimental_preloading']
        self.enable_shuffle = params['enable_shuffle']
        self.batch_size = params['batch_size']
        self.metadata_dtype = params['label_dtype']
        self.data_sharding = False
        if (params['num_slices'] > 1):
            self.data_sharding = True
            self.dataset_num_shards = params['num_slices']
            self.dataset_shard_index = params['slice_index']
            self.data_sharding = True
        self.cycle_length = 10
        self.dataset_parallel_calls = tf.data.experimental.AUTOTUNE  # 64
        self.tf_data_experimental_slack = False
        print("data_dir : ", self.data_dir)
        self.batch_size = fw_params.batch_size

    def __iter__(self):
        self.metadata_dtype_np = get_numpy_dtype(self.metadata_dtype)
        dataset = tf.data.Dataset.list_files(self.data_dir, shuffle=False)
        if self.data_sharding:
            dataset = dataset.shard(self.dataset_num_shards,
                                    self.dataset_shard_index)
            dataset = dataset.shuffle(
                tf.cast(_NUM_TRAIN_FILES / self.dataset_num_shards, tf.int64))
        elif self.enable_shuffle:  # shuffle input files
            dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

        # Prefetch data from files.
        def _prefetch_dataset(filename):
            dataset = tf.data.TFRecordDataset(filename).prefetch(1)
            return dataset

        options = tf.data.Options()
        options.experimental_deterministic = self.enable_experimental_deterministic
        dataset = dataset.interleave(
            map_func=_prefetch_dataset,
            cycle_length=self.cycle_length,
            block_length=1,
            num_parallel_calls=tf.data.experimental.AUTOTUNE).with_options(options)

        if self.enable_cache:  # This part was copied from Coco reader
            # Prefetching and caching increases the memory usage, so disable when
            meminfo = dict((i.split()[0].rstrip(':'), int(i.split()[1]))
                           for i in open('/proc/meminfo').readlines())
            mem_kib = meminfo['MemTotal']
            # rough approx. 1 GiB per tf-record
            caching_mem_kib = len(glob.glob(self.data_dir)) * 1000000
            if caching_mem_kib > mem_kib:
                # performance improved by this
                print(
                    "Dataset cache OFF because MemTotal = %d KiB! It may decrease performance.", mem_kib)
            else:
                dataset = dataset.cache()
                print("Dataset cache ON")

        elif self.enable_shuffle:  # shuffle input images
            dataset = dataset.shuffle(_SHUFFLE_BUFFER)
        # Parse the fetched records to input tensors for model function.
        dataset = dataset.map(
            parse_record, num_parallel_calls=self.dataset_parallel_calls, deterministic=False)

        dataset = dataset.batch(
            batch_size=self.batch_size, drop_remainder=True)
        # dataset = dataset.padded_batch(batch_size=self.batch_size, drop_remainder=True)

        if self.experimental_preloading and len(tf.config.list_logical_devices('HPU')) > 0:
            device = "/device:HPU:0"
            with tf.device(device):
                dataset = dataset.apply(
                    tf.data.experimental.prefetch_to_device(device))
        else:
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
            options = tf.data.Options()
            options.experimental_slack = self.tf_data_experimental_slack
            dataset = dataset.with_options(options)

        self.dataset = dataset
        self.dataset_iterator = iter(self.dataset)
        return self

    def __len__(self):
        # it should return num_batches -- which is unknown
        return 0

    def __next__(self):
        try:
            img_batch, crop_params_batch, label_batch = self.dataset_iterator.get_next()
            img_batch = img_batch.numpy()
            label_batch = label_batch.numpy()
            label_batch = label_batch.astype(self.metadata_dtype_np)
            img_np_buffers = np.empty(
                shape=[self.batch_size, ], dtype=np.object)
            for i in range(self.batch_size):
                img_np_buffers[i] = np.frombuffer(img_batch[i], np.uint8)
                # img_np_buffers[i] = np.pad(
                #    img_np_buffers[i], (0, 64 - len(img_batch[i]) % 64), 'constant')
            return img_np_buffers, label_batch
        except tf.errors.OutOfRangeError:
            raise StopIteration

    def get_media_output_type(self):
        return ro.BUFFER_LIST

    def get_largest_file(self):
        return ""
        # return "/software/data/pytorch/imagenet/ILSVRC2012/train/n03447721/n03447721_43129.JPEG"

    def gen_output_info(self):
        out_info = []
        o = media_ext_reader_op_tensor_info(
            dt.NDT, np.array([self.batch_size], dtype=np.uint32), "")
        out_info.append(o)
        o = media_ext_reader_op_tensor_info(
            self.metadata_dtype, np.array([self.batch_size], dtype=np.uint32), "")
        out_info.append(o)
        return out_info
