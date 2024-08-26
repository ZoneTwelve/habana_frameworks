#!/usr/bin/env python
#
# Copyright 2021-2022 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

from habana_frameworks.mediapipe import fn  # NOQA
from habana_frameworks.mediapipe.mediapipe import MediaPipe  # NOQA
from habana_frameworks.mediapipe.media_types import imgtype as it, randomCropType  # NOQA
from habana_frameworks.mediapipe.media_types import randomCropType as rct  # NOQA
from habana_frameworks.mediapipe.media_types import dtype as dt  # NOQA
from habana_frameworks.mediapipe.operators.cpu_nodes.cpu_nodes import media_function
#from habana_frameworks.mediapipe.plugins.readers.tfrecord_reader import tfr_reader  # NOQA
#from habana_frameworks.mediapipe.plugins.readers.tfrecord_reader import tfr_reader_params  # NOQA
from habana_frameworks.mediapipe.plugins.readers.tfrecord_reader_cpp import tfr_reader_cpp  # NOQA
from habana_frameworks.mediapipe.plugins.readers.tfrecord_reader_cpp import tfr_reader_params  # NOQA


import numpy as np
import math
from PIL import Image
import time


class ResnetPipe(MediaPipe):
    """
    Class defining media resnet pipe.

    """

    def __init__(self, device, queue_depth, batch_size,
                 channel, height, width,
                 is_training, data_dir, out_dtype,
                 num_slices, slice_index, random_crop_type, use_tf_reader=False, dataset_manifest={}):
        """
        Constructor method.

        :params device: device name. <hpu>
        :params queue_depth: queue depth of mediapipe.<1/2/3>
        :params channel: mediapipe image output channel size.
        :params height: mediapipe image output height.
        :params width: mediapipe image output width.
        :params is_training: bool value to state is training pipe or validation pipe.
        :params data_dir: dataset directory to be used by dataset reader.
        :params out_dtype: output image datatype.
        :params num_slices: Total number of slice to be performed on dataset.
        :params slice_index: Slice index to be used for this instance of mediapipe.
        """
        if(is_training == True):
            name = "Train"
        else:
            name = "Eval"

        super(
            ResnetPipe,
            self).__init__(
            device=device,
            prefetch_depth=queue_depth,
            batch_size=batch_size,
            pipe_name=self.__class__.__name__+name)
        self.is_training = is_training
        self.out_dtype = out_dtype
        cast_scale = 1
        cast_zp = 0
        crop_x = 0
        crop_y = 0
        if(out_dtype == dt.FLOAT32 or out_dtype == dt.BFLOAT16):
            cast_scale = 0.03125
            cast_zp = 128
        if(is_training == False):
            crop_x = 0.5
            crop_y = 0.5
        mediapipe_seed = int(time.time_ns() % (2**31 - 1))
        print("media data loader {}/{} seed : {}".format(slice_index,
                                                         num_slices,
                                                         mediapipe_seed))
        print("media data loader: data_dir:", data_dir)
        # file reader  Node
        if(use_tf_reader == True):
            print("TF Reader selected")
            params = tfr_reader_params.copy()
            params['batch_size'] = batch_size
            params['label_dtype'] = dt.FLOAT32
            params['num_slices'] = num_slices
            params['slice_index'] = slice_index
            params['dir'] = data_dir

            if self.is_training:
                params['enable_cache'] = True
                params['enable_experimental_deterministic'] = False
                params['enable_shuffle'] = True
            else:
                params['enable_cache'] = False
                params['enable_experimental_deterministic'] = True
                params['enable_shuffle'] = False

            self.input = fn.MediaExtReaderOp(impl=tfr_reader_cpp,
                                             num_outputs=2,
                                             priv_params=params)
        else:
            print("ReadImageDatasetFromDir selected")

            file_list = dataset_manifest.get('file_list', None)
            class_list = dataset_manifest.get('class_list', None)
            file_sizes = dataset_manifest.get('file_sizes', None)
            file_classes = dataset_manifest.get('file_classes', None)

            self.input = fn.ReadImageDatasetFromDir(dir=data_dir, format="JPEG",
                                                    seed=mediapipe_seed,
                                                    shuffle=True,
                                                    label_dtype=dt.FLOAT32,
                                                    num_slices=num_slices,
                                                    slice_index=slice_index,
                                                    file_list=file_list,
                                                    class_list=class_list,
                                                    file_sizes=file_sizes,
                                                    file_classes=file_classes)
            # max_file="/mnt/weka/data/pytorch/imagenet/ILSVRC2012/train/n03447721/n03447721_43129.JPEG")
        # decoder node
        if(is_training == True):
            priv_params = {}
            priv_params['resize'] = 224
            priv_params['scale'] = [0.08, 1.0]
            priv_params['ratio'] = [3./4., 4./3.]

            self.random_crop = fn.MediaFunc(func=random_crop_func,
                                            shape=[4, batch_size],
                                            dtype=dt.FLOAT32,
                                            seed=mediapipe_seed,
                                            priv_params=priv_params)

            self.decode = fn.ImageDecoder(device="Gaudi2",
                                          output_format=it.RGB_P,
                                          random_crop_type=random_crop_type,
                                          # enable_random_crop=0,
                                          resize=[height, width],
                                          scale_min=0.08,
                                          scale_max=1.0,
                                          ratio_min=3./4.,
                                          ratio_max=4./3.,
                                          seed=mediapipe_seed)
        else:
            self.decode = fn.ImageDecoder(device="Gaudi2",
                                          output_format=it.RGB_P,
                                          random_crop_type=random_crop_type,
                                          resize=[256, 256])

            self.crop = fn.Crop(crop_w=width,
                                crop_h=height,
                                crop_pos_x=crop_x,
                                crop_pos_y=crop_y)
        # Random Flip node
        self.random_flip_input = fn.MediaFunc(func=random_flip_func,
                                              shape=[batch_size],
                                              dtype=dt.UINT8,
                                              seed=mediapipe_seed)

        self.random_flip = fn.RandomFlip(horizontal=1)
        # cast data to f32 for subtraction
        self.cast_pre = fn.Cast(dtype=dt.FLOAT32)
        # substract mean node
        mean_data = np.array([123.68, 116.78, 103.94],
                             dtype=np.float32)

        self.mean_node = fn.MediaConst(data=mean_data,
                                       shape=[1, 1, 3],
                                       dtype=dt.FLOAT32)

        self.sub = fn.Sub(dtype=dt.FLOAT32)
        # cast to output datatype
        self.cast_pst = fn.Cast(dtype=out_dtype)
        # Transpose node
        self.pst_transp = fn.Transpose(permutation=[2, 0, 1, 3],
                                       tensorDim=4,
                                       dtype=out_dtype)

    def definegraph(self):
        """
        Method defining dataflow between nodes.

        :returns : output nodes of the graph defined.
        """
        jpegs, data = self.input()
        if(self.is_training == True):
            # rnd_crop = self.random_crop(jpegs)
            # images = self.decode(jpegs, rnd_crop)
            images = self.decode(jpegs)
            random_flip_input = self.random_flip_input()
            images = self.random_flip(images, random_flip_input)
        else:
            images = self.decode(jpegs)
            images = self.crop(images)
        mean = self.mean_node()
        images = self.cast_pre(images)
        images = self.sub(images, mean)
        # TODO: remove this check once cast to same dtype logic is handled
        # in backend code
        if(self.out_dtype != dt.FLOAT32):
            images = self.cast_pst(images)
        images = self.pst_transp(images)
        return images, data


class random_crop_func(media_function):
    """
    Class defining random crop implementation.

    """

    def __init__(self, params):
        """
        Constructor method.

        :params params: dictionary of params conatining
                        shape: output shape of this class.
                        dtype: output dtype of this class.
                        seed: seed to be used for randomization.
                        priv_params: private params dictionary of this node.
                                    resize: image resize value.
                                    scale: Specifies the lower and upper bounds for the
                                            random area of the crop, before resizing
                                    ratio: lower and upper bounds for the random aspect
                                            ratio of the crop, before resizing.
        """
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.priv_params = params['priv_params']
        self.resize = self.priv_params['resize']
        self.scale = self.priv_params['scale']
        self.ratio = self.priv_params['ratio']
        self.batch_size = self.np_shape[0]
        self.seed = params['seed']
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, filelist):
        """
        Callable class method

        :params filelist: list of images.
        :returns : random crop values calculated per image.
        """
        a = np.empty(shape=self.np_shape, dtype=self.np_dtype)
        for i in range(self.batch_size):
            a[i] = self.random_window_calculator(filelist[i])
        return a

    def random_window_calculator(self, filename):
        """
        Method to calculator random windows for a given image.

        :params filename: image name for which random window needs to be calculated.
        :returns : random crop value generated for given input image.
        """

        clp_value = 48
        clp_value_two_stage = 76
        width, height = Image.open(filename).size
        resize = self.resize
        #print("Image is ",width,height)
        area = width*height
        # print(area)
        scale = np.array([self.scale[0], self.scale[1]]
                         )  # np.array([0.08,1.0])
        ratio = np.array([self.ratio[0], self.ratio[1]]
                         )  # np.array([3./4.,4./3.])
        # log_ratio = torch.log(torch.tensor(ratio))
        log_ratio = np.log(ratio)
        for _ in range(10):
            #target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            target_area = area * self.rng.uniform(scale[0], scale[1])
            # aspect_ratio = torch.exp(
            #    torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            # ).item()
            aspect_ratio = math.exp(
                self.rng.uniform(log_ratio[0], log_ratio[1]))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            w = max(w, clp_value)
            h = max(h, clp_value)
            if((w < resize and h > resize) or (w > resize and h < resize)):
                w = max(w, clp_value_two_stage)
                h = max(h, clp_value_two_stage)
            w = min(w, width)
            h = min(h, height)
            if 0 < w <= width and 0 < h <= height:
                #i = torch.randint(0, height - h + 1, size=(1,)).item()
                #j = torch.randint(0, width - w + 1, size=(1,)).item()
                i = self.rng.integers(0, width - w + 1)
                j = self.rng.integers(0, height - h + 1)
                return [i/width, j/height, w/width, h/height]

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        w = max(w, clp_value)
        h = max(h, clp_value)
        if((w < resize and h > resize) or (w > resize and h < resize)):
            w = max(w, clp_value_two_stage)
            h = max(h, clp_value_two_stage)
        w = min(w, width)
        h = min(h, height)
        i = (width - w) // 2
        j = (height - h) // 2
        # return i, j, h, w
        return [i/width, j/height, w/width, h/height]


class random_crop_func_1(media_function):
    """
    Class defining random crop implementation.

    """

    def __init__(self, params):
        """
        Constructor method.

        :params params: dictionary of params conatining
                        shape: output shape of this class.
                        dtype: output dtype of this class.
                        seed: seed to be used for randomization.
        """
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.batch_size = self.np_shape[0]
        self.seed = params['seed']
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, filelist):
        """
        Callable class method.

        :params filelist: list of images.
        :returns : random crop values calculated per image.
        """
        a = np.empty(shape=self.np_shape, dtype=self.np_dtype)
        x_val = self.rng.uniform(0, .2, self.batch_size)
        y_val = self.rng.uniform(0, .3, self.batch_size)
        w_val = self.rng.uniform(0.8, 1, self.batch_size)
        h_val = self.rng.uniform(0.7, 1, self.batch_size)
        for i in range(self.batch_size):
            if((x_val[i] + w_val[i]) > 1):
                w_val[i] = 1 - x_val[i]
            if((y_val[i] + h_val[i]) > 1):
                h_val[i] = 1 - y_val[i]
            a[i] = [x_val[i], y_val[i], w_val[i], h_val[i]]
        return a


class random_flip_func(media_function):
    """
    Class defining the random flip implementation.

    """

    def __init__(self, params):
        """
        Constructor method.

        :params params: dictionary of params conatining
                        shape: output shape of this class.
                        dtype: output dtype of this class.
                        seed: seed to be used for randomization.
        """
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.seed = params['seed']
        self.rng = np.random.default_rng(self.seed)

    def __call__(self):
        """
        Callable class method

        :returns : random flip values calculated per image.
        """
        a = self.rng.choice([0, 1], p=[0.5, 0.5], size=self.np_shape)
        a = np.array(a, dtype=self.np_dtype)
        return a
