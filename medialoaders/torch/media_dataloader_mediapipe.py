from habana_frameworks.mediapipe.mediapipe import MediaPipe  # NOQA
from habana_frameworks.mediapipe import fn  # NOQA
from habana_frameworks.mediapipe.media_types import imgtype as it  # NOQA
from habana_frameworks.mediapipe.media_types import dtype as dt  # NOQA
from habana_frameworks.mediapipe.media_types import ftype as ft  # NOQA
from habana_frameworks.mediapipe.media_types import layout as lt  # NOQA
from habana_frameworks.mediapipe.media_types import randomCropType as rct  # NOQA
from habana_frameworks.mediapipe.media_types import decoderStage as ds  # NOQA
from habana_frameworks.mediapipe.operators.cpu_nodes.cpu_nodes import media_function  # NOQA
from media_pipe_api import MetadataOps

import os
import time
import sys

import torch
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import copy
import math
#from datetime import datetime
from PIL import Image
#from collections.abc import Sequence
from enum import Enum
from enum import IntEnum


class Crop_Type(Enum):
    """
    Enum class defining crop-resize type.

    """
    Crop_Only = 0
    ResizedCrop = 1
    Crop_Resize = 2
    Resize_Crop = 3
    Ssd_Crop = 4


class Additional_Transform_Type(IntEnum):
    """
    Enum class defining additional transforms for ssd

    """
    SSD_Crop = 0
    hflip = 1
    normalize = 2


class random_flip_func(media_function):
    """
    Class to randomly generate input for RandomFlip media node.

    """

    def __init__(self, params):
        """
        Constructor method.

        :params params: random_flip_func specific params.
                        shape: output shape
                        dtype: output data type
                        seed: seed to be used
        """
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.seed = params['seed']
        self.rng = np.random.default_rng(self.seed)

    def __call__(self):
        """
        Callable class method.

        :returns : randomly generated binary output per image.
        """
        a = self.rng.choice([0, 1], p=[0.5, 0.5], size=self.np_shape)
        a = np.array(a, dtype=self.np_dtype)
        return a


class random_resized_crop_func(media_function):
    """
    Class to generate random crop parameters.

    """

    def __init__(self, params):
        """
        Constructor method.

        :params params: random_resized_crop_func specific params
                        shape: output shape.
                        dtype: output data type.
                        seed: seed to be used.
                        priv_params: private params for random_resized_crop_func
                                     resizewidth: resize output width.
                                     resizeheight: resize output height.
                                     scale: lower and upper bounds for the random area of the crop, before resizing.
                                     ratio: lower and upper bounds for the random aspect ratio of the crop, before resizing.
        """
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.batch_size = self.np_shape[0]
        self.priv_params = params['priv_params']
        self.resizeWidth = self.priv_params['resizewidth']
        self.resizeHeight = self.priv_params['resizeheight']
        self.scale = self.priv_params['scale']
        self.ratio = self.priv_params['ratio']
        self.seed = params['seed']
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, filelist):
        """
        Callable class method.

        :params filelist: batch of files.
        :returns : random crop parameters for filelist.
        """
        a = np.zeros(shape=self.np_shape, dtype=self.np_dtype)
        for i in range(self.batch_size):
            a[i] = self.random_window_calculator(filelist[i])
        # print("RandomResizedCrop: ", a)
        return a

    def random_window_calculator(self, filename):
        """
        Method to generate crop params for a file.

        :params filename: file for which crop params are to be generated.
        :returns : random crop parameters for specified file.
        """
        clp_value = 48
        clp_value_two_stage = 76
        width, height = Image.open(filename).size
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
            if((w < self.resizeWidth and h > self.resizeHeight) or (w > self.resizeWidth and h < self.resizeHeight)):
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
        if((w < self.resizeWidth and h > self.resizeHeight) or (w > self.resizeWidth and h < self.resizeHeight)):
            w = max(w, clp_value_two_stage)
            h = max(h, clp_value_two_stage)
        w = min(w, width)
        h = min(h, height)

        i = (width - w) // 2
        j = (height - h) // 2
        # return i, j, h, w
        return [i/width, j/height, w/width, h/height]


class center_crop_func(media_function):
    """
    Class to generate center crop parameters.

    """

    def __init__(self, params):
        """
        Constructor method.

        :params params: center_crop_func specific params.
                        shape: output shape
                        dtype: output data type
                        priv_params: private params for center_crop_func
                                     cropWidth: crop output width
                                     cropHeight: crop output height
        """
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.batch_size = self.np_shape[0]
        self.priv_params = params['priv_params']
        self.cropWidth = self.priv_params['cropWidth']
        self.cropHeight = self.priv_params['cropHeight']

    def __call__(self, filelist):
        """
        Callable class method.

        :params filelist: batch of files.
        :returns : center crop parameters for filelist.
        """
        a = np.zeros(shape=self.np_shape, dtype=self.np_dtype)
        i = 0
        for filename in filelist:
            width, height = Image.open(filename).size
            # print("Image is ",filename, width,height)
            if width > self.cropWidth:
                crop_x = (width - self.cropWidth) // 2
                crop_x_ratio = crop_x / width
                crop_w_ratio = self.cropWidth / width
            else:
                crop_x_ratio = 0
                crop_w_ratio = 1

            if height > self.cropHeight:
                crop_y = (height - self.cropHeight) // 2
                crop_y_ratio = crop_y / height
                crop_h_ratio = self.cropHeight / height
            else:
                crop_y_ratio = 0
                crop_h_ratio = 1
            a[i] = [crop_x_ratio, crop_y_ratio, crop_w_ratio, crop_h_ratio]
            i += 1
        # print("CenterCrop: ", a)
        return a


class random_brigtness_func(media_function):
    """
    Class to randomly generate input for brightness media node.

    """

    def __init__(self, params):
        """
        Constructor method.

        :params params: random_brigtness_func specific params
                        shape: output shape.
                        dtype: output data type.
                        seed: seed to be used.
                        priv_params: private params for random_brigtness_func
                                     range: range of values to be generated
        """
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.priv_params = params['priv_params']
        self.range = self.priv_params['range']
        self.min = self.range[0]  # 0.875
        self.max = self.range[1]  # 1.125
        self.seed = params['seed']
        self.rng = np.random.default_rng(int(self.seed + 10))
        #print("brightness range ", self.min, self.max)

    def __call__(self):
        """
        Callable class method.

        :returns : randomly generated output per image.
        """
        rnd_data = self.rng.uniform(
            low=self.min, high=self.max, size=self.np_shape)
        a = np.array(rnd_data, dtype=self.np_dtype)
        return a


class random_contrast_func(media_function):
    """
    Class to randomly generate input for contrast media node.

    """

    def __init__(self, params):
        """
        Constructor method.

        :params params: random_contrast_func specific params
                        shape: output shape.
                        dtype: output data type.
                        seed: seed to be used.
                        priv_params: private params for random_contrast_func
                                     range: range of values to be generated
        """
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.priv_params = params['priv_params']
        self.range = self.priv_params['range']
        self.min = self.range[0]  # 0.5
        self.max = self.range[1]  # 1.5
        self.seed = params['seed']
        self.rng = np.random.default_rng(int(self.seed + 20))
        #print("contrast range ", self.min, self.max)

    def __call__(self):
        """
        Callable class method.

        :returns : randomly generated output per image.
        """
        rnd_data = self.rng.uniform(
            low=self.min, high=self.max, size=self.np_shape)
        a = np.array(rnd_data, dtype=self.np_dtype)
        return a


class random_saturation_func(media_function):
    """
    Class to randomly generate input for saturation media node.

    """

    def __init__(self, params):
        """
        Constructor method.

        :params params: random_saturation_func specific params
                        shape: output shape.
                        dtype: output data type.
                        seed: seed to be used.
                        priv_params: private params for random_saturation_func
                                     range: range of values to be generated
        """
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.priv_params = params['priv_params']
        self.range = self.priv_params['range']
        self.min = self.range[0]  # 0.5
        self.max = self.range[1]  # 1.5
        self.seed = params['seed']
        self.rng = np.random.default_rng(int(self.seed + 30))
        #print("saturation range ", self.min, self.max)

    def __call__(self):
        """
        Callable class method.

        :returns : randomly generated output per image.
        """
        rnd_data = self.rng.uniform(
            low=self.min, high=self.max, size=self.np_shape)
        a = np.array(rnd_data, dtype=self.np_dtype)
        return a


class random_hue_func(media_function):
    """
    Class to randomly generate input for hue media node.

    """

    def __init__(self, params):
        """
        Constructor method.

        :params params: random_hue_func specific params
                        shape: output shape.
                        dtype: output data type.
                        seed: seed to be used.
                        priv_params: private params for random_hue_func
                                     range: range of values to be generated
        """
        self.np_shape = params['shape'][::-1]
        self.batch_size = self.np_shape[0]
        self.np_dtype = params['dtype']
        self.priv_params = params['priv_params']
        self.range = self.priv_params['range']
        self.min = self.range[0]  # -0.05
        self.max = self.range[1]  # 0.05
        self.seed = params['seed']
        self.rng = np.random.default_rng(int(self.seed + 40))
        #print("hue range ", self.min, self.max)

    def __call__(self):
        """
        Callable class method.

        :returns : randomly generated output per image.
        """
        rnd_data = self.rng.uniform(
            low=self.min, high=self.max, size=self.np_shape)
        rnd_data = rnd_data * 360
        for x in range(self.batch_size):
            #rnd_data[x] = rnd_data[x] * 360
            if rnd_data[x] < 0:
                rnd_data[x] = 360 + rnd_data[x]
        a = np.array(rnd_data, dtype=self.np_dtype)
        return a


class HPUMediaPipe(MediaPipe):
    """
    Class defining resnet media pipe.

    """
    instance_count = 0

    def __init__(self, a_torch_transforms=None, a_root=None, a_annotation_file=None, a_batch_size=1, a_shuffle=False, a_drop_last=True, a_prefetch_count=1, a_num_instances=1, a_instance_id=0, a_model_ssd=False, a_device=None, a_dataset_manifest={}):
        """
        Constructor method.

        :params a_torch_transforms: transforms to be applied on mediapipe.
        :params a_root: path from which to load the images.
        :params a_annotation_file: path from which to load annotation file for SSD.
        :params a_batch_size: mediapipe output batch size.
        :params a_shuffle: whether images have to be shuffled.
        :params a_drop_last: whether to drop the last incomplete batch or round up.
        :params a_prefetch_count: queue depth for media processing. <1/2/3>
        :params a_num_instances: number of devices.
        :params a_instance_id: instance id of current device.
        :params a_model_ssd: whether mediapipe is to be created for SSD.
        :params a_device: media device to run mediapipe on. <hpu/hpu:0>
        :params a_dataset_manifest: dictionary describing dataset for Resnet.
        """
        self.super_init = False
        self.root = a_root
        batchSize = a_batch_size
        self.shuffle = a_shuffle  # ToDo: Update shuffle for train/val for SSD
        self.drop_last = a_drop_last
        self.transform_to_ignore = []
        self.crop_transform_index = None
        self.resize_transform_index = None
        self.cmn_transform_index = None
        self.dataset_manifest = a_dataset_manifest

        num_crop = 0
        num_resize = 0
        num_color_jitter = 0
        num_cmn = 0
        crop_width = None
        crop_height = None
        resize_width = None
        resize_height = None

        self.crop_type = None
        resize_crop_handled = True
        self.crop_width = None
        self.crop_height = None
        self.need_crop_op = False
        self.decode_width = None
        self.decode_height = None
        self.media_output_dtype = 'uint8'
        self.annotation_file = a_annotation_file
        num_ssd_horizontal_flip = 0
        ssd_crop = False
        self.additional = []
        self.ssd_train = None
        self.ssd_num_cropping_iterations = None
        if not isinstance(a_model_ssd, bool):
            raise ValueError("Unsupported value of a_model_ssd ", a_model_ssd)
        self.model_ssd = a_model_ssd

        if self.model_ssd == True:
            transform = a_torch_transforms
            if not isinstance(transform.val, bool):
                raise ValueError(
                    "Unsupported value of transform.val ", transform.val)
            self.ssd_train = not transform.val

            if self.ssd_train == True:
                print("SSD train HPUMediaPipe")
                if not isinstance(transform.img_trans, transforms.Compose):
                    raise ValueError(
                        "transform.img_trans should be of type torchvision.transforms")

                self.transforms = transform.img_trans.transforms
                if (not hasattr(transform, "crop")) or (not hasattr(transform, "hflip")) or (not hasattr(transform, "normalize")):
                    raise ValueError(
                        "crop, hflip, normalize needed for SSD train media pipe")
                self.additional = [transform.crop,
                                   transform.hflip, transform.normalize]
            else:
                print("SSD val HPUMediaPipe")
                if not isinstance(transform.trans_val, transforms.Compose):
                    raise ValueError(
                        "transform.trans_val should be of type torchvision.transforms")

                self.transforms = transform.trans_val.transforms
                self.additional = []

            # print("SSD transforms are ", self.transforms + self.additional)
        else:
            if not isinstance(a_torch_transforms, transforms.Compose):
                raise ValueError(
                    "torch_transforms should be of type torchvision.transforms")
            self.transforms = a_torch_transforms.transforms
            # print("transforms are ", self.transforms)

        transform_count = 0

        for t in (self.transforms + self.additional):
            if isinstance(t, transforms.RandomResizedCrop):
                if self.model_ssd == True:
                    raise ValueError(
                        "Unsupported transform for SSD: " + str(type(t)))
                if num_crop == 0 and num_resize == 0:
                    print("transform RandomResizedCrop: Random Crop,Resize w:h ",
                          t.size[1], t.size[0], " scale: ", t.scale, " ratio: ", t.ratio, " interpolation: ", t.interpolation)
                    num_crop += 1
                    num_resize += 1
                else:
                    raise ValueError(
                        "Unsupported 2nd crop/resize transform: " + str(type(t)))
                resize_height = t.size[0]
                resize_width = t.size[1]
                if (resize_width % 2 != 0) or (resize_height % 2 != 0):  # ToDo: check
                    raise ValueError(
                        "Unsupported w:h for transform: " + str(type(t)))
                self.crop_type = Crop_Type.ResizedCrop
                self.transform_to_ignore.append(transform_count)
                self.crop_transform_index = transform_count
                self.resize_transform_index = transform_count

            elif isinstance(t, transforms.CenterCrop):
                if self.model_ssd == True:
                    raise ValueError(
                        "Unsupported transform for SSD: " + str(type(t)))
                if num_crop == 0:
                    print("transform CenterCrop: w:h ", t.size[1], t.size[0])
                    num_crop += 1
                else:
                    raise ValueError(
                        "Unsupported 2nd Crop transform: " + str(type(t)))

                crop_height = t.size[0]
                crop_width = t.size[1]

                if num_resize == 0:
                    self.crop_type = Crop_Type.Crop_Only
                    self.transform_to_ignore.append(transform_count)
                else:
                    self.crop_type = Crop_Type.Resize_Crop
                    if num_cmn == 0:
                        resize_crop_handled = False
                    else:
                        resize_crop_handled = True

                self.crop_transform_index = transform_count

            elif isinstance(t, transforms.Resize):
                check_maxsize = False
                if num_resize == 0:
                    if isinstance(t.size, int):
                        resize_height = t.size
                        resize_width = t.size
                        check_maxsize = True
                    elif (isinstance(t.size, tuple) or (isinstance(t.size, list))) and len(t.size) == 1:
                        resize_height = t.size[0]
                        resize_width = t.size[0]
                        check_maxsize = True
                    elif (isinstance(t.size, tuple) or (isinstance(t.size, list))) and len(t.size) == 2:
                        resize_height = t.size[0]
                        resize_width = t.size[1]
                    else:
                        raise ValueError(
                            "Unsupported size: transforms.Resize ")
                    print("transform Resize: w:h ", resize_width, resize_height, " interpolation: ",
                          t.interpolation, " max_size: ", t.max_size)  # t.antialias ignored

                    num_resize += 1

                    if (resize_width % 2 != 0) or (resize_height % 2 != 0):  # ToDo: Check
                        raise ValueError(
                            "Unsupported w:h for transform: " + str(type(t)))

                    # resize_height will be same as resize_width
                    if (check_maxsize == True) and (t.max_size != None) and (resize_width > t.max_size):
                        raise ValueError(
                            "max_size must be greater than resize width/height for transform: " + str(type(t)))

                else:
                    raise ValueError(
                        "Unsupported 2nd resize transform: " + str(type(t)))

                # Update crop type to Crop_Resize
                if (num_crop == 1) and (self.crop_type == Crop_Type.Crop_Only):
                    self.crop_type = Crop_Type.Crop_Resize

                self.transform_to_ignore.append(transform_count)
                self.resize_transform_index = transform_count

            elif isinstance(t, transforms.Normalize):
                if num_cmn == 0:
                    print("transform Normalize: mean:std",
                          t.mean, t.std)  # t.inplace ignored

                    if ((not isinstance(t.mean, list)) and (not isinstance(t.mean, tuple))) or (len(t.mean) != 3):
                        raise ValueError(
                            "Unsupported mean for Normalize transform")

                    if ((not isinstance(t.std, list)) and (not isinstance(t.std, tuple))) or (len(t.std) != 3):
                        raise ValueError(
                            "Unsupported std for Normalize transform")

                    if t.mean != [0.485, 0.456, 0.406] and t.mean != (0.485, 0.456, 0.406):
                        print(
                            "transform Normalize mean is different than default mean")

                    if t.std != [0.229, 0.224, 0.225] and t.std != (0.229, 0.224, 0.225):
                        print("transform Normalize std is different than default std")
                    num_cmn += 1
                else:
                    raise ValueError(
                        "Unsupported 2nd Normalize transform: " + str(type(t)))

                if resize_crop_handled == False:
                    resize_crop_handled = True

                # cmn to be added as last node in media graph
                if t in self.transforms:
                    if self.ssd_train == True:  # Normalize transform is supported with self.additional for SSD Train
                        raise ValueError(
                            "Unsupported Normalize transform for SSD Train: " + str(type(t)))
                    self.cmn_transform_index = transform_count
                    self.transform_to_ignore.append(transform_count)

            elif isinstance(t, transforms.RandomHorizontalFlip):
                if self.model_ssd == True:  # expected utils.RandomHorizontalFlip
                    raise ValueError(
                        "Unsupported transform for SSD: " + str(type(t)))
                if t.p == 0.5:
                    print("transform RandomHorizontalFlip: probability ", t.p)
                else:
                    raise ValueError(
                        "Unsupported probability for transform: " + str(type(t)))  # TODO

            elif isinstance(t, transforms.RandomVerticalFlip):
                if self.model_ssd == True:
                    raise ValueError(
                        "Unsupported transform for SSD: " + str(type(t)))
                if t.p == 0.5:
                    print("transform RandomVerticalFlip: probability ", t.p)
                else:
                    raise ValueError(
                        "Unsupported probability for transform: " + str(type(t)))  # TODO

            elif isinstance(t, transforms.ToTensor):
                print("transform ToTensor")
                self.media_output_dtype = 'float32'  # TODO
                self.transform_to_ignore.append(transform_count)

            elif isinstance(t, transforms.ColorJitter):
                if num_color_jitter == 0:  # ToDo: check if > 1 ColorJitter can be supported
                    if t.brightness is None:
                        print("Brightness not available in ColorJitter")
                    else:
                        if (not isinstance(t.brightness, (tuple, list))) or (len(t.brightness) != 2):
                            raise ValueError(
                                "Unsupported brightness for ColorJitter transform")
                    if t.contrast is None:
                        print("Contrast not available in ColorJitter")
                    else:
                        if (not isinstance(t.contrast, (tuple, list))) or (len(t.contrast) != 2):
                            raise ValueError(
                                "Unsupported contrast for ColorJitter transform")
                    if t.saturation is None:
                        print("Saturation not available in ColorJitter")
                    else:
                        if (not isinstance(t.saturation, (tuple, list))) or (len(t.saturation) != 2):
                            raise ValueError(
                                "Unsupported saturation for ColorJitter transform")
                    if t.hue is None:
                        print("Hue not available in ColorJitter")
                    else:
                        if (not isinstance(t.hue, (tuple, list))) or (len(t.hue) != 2):
                            raise ValueError(
                                "Unsupported hue for ColorJitter transform")

                    print("transform ColorJitter: brightness {} contrast {} saturation {} hue {} ".format(
                        t.brightness, t.contrast, t.saturation, t.hue))
                    num_color_jitter += 1
                else:
                    raise ValueError(
                        "Unsupported 2nd color jitter transform: " + str(type(t)))

            # WA for SSD transforms that are privately defined in model garden
            elif 'utils.SSDCropping' in str(type(t)):
                # SSDCropping supported only for SSD train
                if (self.model_ssd == False) or (self.ssd_train == False):
                    raise ValueError("Unsupported transform : " + str(type(t)))

                if t.num_cropping_iterations < 1:
                    raise ValueError(
                        "Unsupported num_cropping_iterations for transform : " + str(type(t)))

                print("transform SSDCropping: num_cropping_iterations ",
                      t.num_cropping_iterations)  # t.sample_options
                self.ssd_num_cropping_iterations = t.num_cropping_iterations

                if num_crop == 0:
                    num_crop += 1
                else:
                    raise ValueError(
                        "Unsupported 2nd Crop transform: " + str(type(t)))

                if num_resize == 0:  # transforms.Resize should be part of self.transforms for SSD train
                    raise ValueError(
                        "Unsupported SSD Crop transform without resize")

                self.crop_type = Crop_Type.Ssd_Crop
                ssd_crop = True

            elif 'utils.RandomHorizontalFlip' in str(type(t)):
                if (self.model_ssd == False) or (self.ssd_train == False):
                    raise ValueError("Unsupported transform : " + str(type(t)))

                if num_ssd_horizontal_flip == 0:
                    if t.p == 0.5:
                        print(
                            "transform utils.RandomHorizontalFlip: probability ", t.p)
                    else:
                        raise ValueError(
                            "Unsupported probability for transform: " + str(type(t)))  # TODO
                    num_ssd_horizontal_flip += 1
                else:
                    raise ValueError(
                        "Unsupported 2nd horizontal flip transform : " + str(type(t)))

            else:
                raise ValueError("Unsupported transform: " + str(type(t)))

            transform_count += 1

        if (num_cmn == 0) and (self.media_output_dtype == 'float32'):
            raise ValueError("Unsupported output data type float32")  # TODO

        if self.model_ssd == True:
            if (self.ssd_train == True) and (num_ssd_horizontal_flip != 1):
                raise ValueError(
                    "horizontal flip transform needed for SSD train media pipe")
            if (self.ssd_train == True) and (ssd_crop == False):
                raise ValueError(
                    "ssd crop transform needed for SSD train media pipe")
            if (self.ssd_train == False) and (ssd_crop == True):
                raise ValueError(
                    "ssd crop transform not supported for SSD val media pipe")
            # if (a_num_instances != 1) or (a_instance_id != 0):  # ToDo update
            #    raise ValueError(
            #        "Only a_num_instances 1 and a_instance_id 0 supported for SSD media pipe")

        if num_resize == 1:
            self.decode_width = resize_width
            self.decode_height = resize_height
        elif num_resize == 0:
            if self.crop_type == Crop_Type.Crop_Only:
                if (crop_width % 2 != 0) or (crop_height % 2 != 0):  # ToDo: Check
                    raise ValueError(
                        "Unsupported w:h for transform: transforms.CenterCrop")
                self.decode_width = crop_width
                self.decode_height = crop_height
                print("No resize found, using crop resolution of ",
                      self.decode_width, "x", self.decode_height)
            else:
                raise ValueError("No resize/crop found")
                #self.decode_width = default_output_size
                #self.decode_height = default_output_size
        else:
            raise ValueError("Unsupported resize count")

        if num_crop == 1:
            if self.crop_type == Crop_Type.Resize_Crop:
                self.enable_crop = False
                if (resize_width < crop_width) or (resize_height < crop_height):
                    print(" resize width:height ", resize_width, resize_height,
                          "crop width:height ", crop_width, crop_height)
                    raise ValueError(
                        "Unsupported crop width/height > resize width/height")
            else:
                self.enable_crop = True
                if (self.crop_type == Crop_Type.Crop_Only) or (self.crop_type == Crop_Type.Crop_Resize):
                    self.crop_width = crop_width
                    self.crop_height = crop_height
                # elif (self.crop_type == Crop_Type.ResizedCrop) or (self.crop_type == Crop_Type.Ssd_Crop):

        else:
            if num_crop != 0:
                raise ValueError("Unsupported crop count")
            self.enable_crop = False

        if resize_crop_handled == False:
            self.need_crop_op = True

        # np.random.seed(int(time.time()))
        # np.random.seed(1000)

        self.num_instances = a_num_instances
        self.instance_id = a_instance_id

        print("MediaDataloader num instances {} instance id {}".format(
            self.num_instances, self.instance_id))

        HPUMediaPipe.instance_count += 1
        pipename = "{}:{}".format(
            self.__class__.__name__, HPUMediaPipe.instance_count)
        pipename = str(pipename)

        self.super_init = True
        super().__init__(device=a_device, batch_size=batchSize,
                         prefetch_depth=a_prefetch_count, pipe_name=pipename)

    def __del__(self):
        """
        Destructor method.

        """
        if self.super_init == True:
            super().__del__()

    def definegraph(self):
        """
        Method defines the media graph based on transforms.

        :returns : output images, labels
        """
        # seed_mediapipe = 1000 #TODO: Update
        seed_mediapipe = int(time.time_ns() % (2**31 - 1))

        # try to print the seed for distributed
        try:
            print("MediaDataloader {}/{} seed : {}".format(self.instance_id,
                  self.num_instances, seed_mediapipe), force=True)
        except TypeError:
            print("MediaDataloader seed : {}".format(seed_mediapipe))

        crop_func = None
        enable_decoder_random_crop = False
        batchSize = self.getBatchSize()

        if self.enable_crop == True:
            if self.crop_type == Crop_Type.ResizedCrop:

                # random_crop_params = {}
                # random_crop_params['resizewidth'] = self.decode_width
                # random_crop_params['resizeheight'] = self.decode_height
                # random_crop_params['scale'] = self.transforms[self.crop_transform_index].scale
                # random_crop_params['ratio'] = self.transforms[self.crop_transform_index].ratio
                # crop_func = random_resized_crop_func
                enable_decoder_random_crop = True
                crop_func = None
                print("Decode ResizedCrop w:h",
                      self.decode_width, self.decode_height)

            elif (self.crop_type == Crop_Type.Ssd_Crop):
                print("Decode w:h {} {} with SSD crop".format(
                    self.decode_width, self.decode_height))

            elif (self.crop_type == Crop_Type.Crop_Only) or (self.crop_type == Crop_Type.Crop_Resize):
                random_crop_params = {}
                random_crop_params['cropWidth'] = self.crop_width
                random_crop_params['cropHeight'] = self.crop_height
                crop_func = center_crop_func

                print("Decode w:h ", self.decode_width, self.decode_height,
                      "Center Crop w:h: ", self.crop_width, self.crop_height)

            else:
                assert False, "Error: wrong crop type"
        else:
            # No Crop or crop_type = Crop_Type.Resize_Crop
            print("Decode w:h ", self.decode_width,
                  self.decode_height, " , Crop disabled")
            crop_func = None

        if self.resize_transform_index == None:
            assert self.crop_type == Crop_Type.Crop_Only, "No resize transform available"
            res_pp_filter = ft.BI_LINEAR  # 3
        # Force pp filter to bicubic for SSD eval
        # elif self.ssd_train == False:  # ToDo:
        #    res_pp_filter = ft.BICUBIC  # 4
        else:
            if self.transforms[self.resize_transform_index].interpolation == InterpolationMode.BILINEAR:
                res_pp_filter = ft.BI_LINEAR  # 3
            elif self.transforms[self.resize_transform_index].interpolation == InterpolationMode.NEAREST:
                res_pp_filter = ft.NEAREST  # 2
            elif self.transforms[self.resize_transform_index].interpolation == InterpolationMode.BICUBIC:
                res_pp_filter = ft.BICUBIC  # 4
            elif self.transforms[self.resize_transform_index].interpolation == InterpolationMode.BOX:
                res_pp_filter = ft.BOX  # 6
            elif self.transforms[self.resize_transform_index].interpolation == InterpolationMode.LANCZOS:
                res_pp_filter = ft.LANCZOS  # 1
            elif self.transforms[self.resize_transform_index].interpolation == InterpolationMode.HAMMING:
                print(
                    "Warning: InterpolationMode.HAMMING not supported, using InterpolationMode.BILINEAR")
                res_pp_filter = ft.BI_LINEAR
            else:
                assert False, "Error: Unsupported InterpolationMode"

        # if crop_func == None:
        #    print("Decode w:h ", self.decode_width, self.decode_height, " with Crop Disabled")
        print("MediaDataloader shuffle is ", self.shuffle)
        print("MediaDataloader output type is ", self.media_output_dtype)

        if self.model_ssd == True:
            if self.ssd_train == True:
                output_partial = False  # ToDo: check if need to output partial batch for train
            else:
                output_partial = True
            self.input = fn.CocoReader(root=self.root, annfile=self.annotation_file, seed=seed_mediapipe, shuffle=self.shuffle,
                                       drop_remainder=self.drop_last, num_slices=self.num_instances, slice_index=self.instance_id, partial_batch=output_partial)
            jpegs, ids, sizes, boxes, labels, lengths, batch = self.input()
        else:
            file_list = None
            class_list = None
            file_sizes = None
            file_classes = None
            if self.dataset_manifest:
                file_list = self.dataset_manifest.get('file_list', None)
                class_list = self.dataset_manifest.get('class_list', None)
                file_sizes = self.dataset_manifest.get('file_sizes', None)
                file_classes = self.dataset_manifest.get('file_classes', None)

            self.input = fn.ReadImageDatasetFromDir(dir=self.root, format=["jpg", "JPG", "jpeg", "JPEG"],  # "JPEG"
                                                    seed=seed_mediapipe,
                                                    shuffle=self.shuffle,
                                                    drop_remainder=self.drop_last,
                                                    label_dtype=dt.UINT32,
                                                    num_slices=self.num_instances,
                                                    slice_index=self.instance_id,
                                                    file_list=file_list,
                                                    class_list=class_list,
                                                    file_sizes=file_sizes,
                                                    file_classes=file_classes)  # TODO: check label_dtype
            jpegs, data = self.input()

        def_output_image_size = [self.decode_width, self.decode_height]
        decode_stage = ds.ENABLE_ALL_STAGES
        if enable_decoder_random_crop == True:
            # crop_type = Crop_Type.ResizedCrop

            self.decode = fn.ImageDecoder(output_format=it.RGB_P, resize=def_output_image_size, resampling_mode=res_pp_filter, random_crop_type=rct.RANDOMIZED_AREA_AND_ASPECT_RATIO_CROP,
                                          scale_min=self.transforms[self.crop_transform_index].scale[0], scale_max=self.transforms[self.crop_transform_index].scale[1], ratio_min=self.transforms[self.crop_transform_index].ratio[0], ratio_max=self.transforms[self.crop_transform_index].ratio[1], seed=seed_mediapipe, decoder_stage=decode_stage)
            images = self.decode(jpegs)
        else:
            self.decode = fn.ImageDecoder(
                output_format=it.RGB_P, resize=def_output_image_size, resampling_mode=res_pp_filter, decoder_stage=decode_stage)

            if crop_func is not None:
                # crop_func is center_crop_func or random_resized_crop_func
                self.random_crop = fn.MediaFunc(func=crop_func, dtype=dt.FLOAT32, shape=[
                                                4, batchSize], priv_params=random_crop_params, seed=seed_mediapipe)
                crop_val = self.random_crop(jpegs)
                images = self.decode(jpegs, crop_val)
            elif self.crop_type == Crop_Type.Ssd_Crop:
                # crop_type is Ssd_Crop, so SSD hflip should also be available
                # transform_ssd = self.additional[Additional_Transform_Type.SSD_Crop]
                # transform_hflip = self.additional[Additional_Transform_Type.hflip]
                self.ssd_metadata = fn.SSDMetadata(
                    workers=1, serialize=[MetadataOps.crop, MetadataOps.flip, MetadataOps.encode], cropping_iterations=self.ssd_num_cropping_iterations, seed=seed_mediapipe)
                self.random_flip_input = fn.MediaFunc(func=random_flip_func, shape=[
                                                      batchSize], dtype=dt.UINT8, seed=seed_mediapipe)
                self.random_flip = fn.RandomFlip(horizontal=1)

                flip = self.random_flip_input()
                windows, ids, sizes, boxes, labels, lengths = self.ssd_metadata(
                    ids, sizes, boxes, labels, lengths, flip)
                images = self.decode(jpegs, windows)
                images = self.random_flip(images, flip)

            else:
                images = self.decode(jpegs)

        width = def_output_image_size[0]
        height = def_output_image_size[1]
        crop_cmn = False
        transform_count = 0
        for t in self.transforms:
            if transform_count in self.transform_to_ignore:
                #print("ignored transform ", str(type(t)))
                pass
            else:
                if isinstance(t, transforms.CenterCrop):
                    assert self.crop_type == Crop_Type.Resize_Crop, "Wrong Crop Type"
                    height = t.size[0]
                    width = t.size[1]
                    cmn_pos_offset = 0.5
                    if self.need_crop_op == True:
                        self.crop_op = fn.Crop(
                            crop_w=width, crop_h=height, crop_pos_x=cmn_pos_offset, crop_pos_y=cmn_pos_offset, crop_d=0)
                        images = self.crop_op(images)
                    else:
                        crop_cmn = True

                elif isinstance(t, transforms.RandomHorizontalFlip):
                    self.random_flip_input = fn.MediaFunc(
                        func=random_flip_func, shape=[batchSize], dtype=dt.UINT8, seed=seed_mediapipe)
                    self.random_flip = fn.RandomFlip(horizontal=1)
                    flip = self.random_flip_input()
                    images = self.random_flip(images, flip)

                elif isinstance(t, transforms.RandomVerticalFlip):
                    # random_flip_fn = rng_for_flip_function
                    self.random_flip_input = fn.MediaFunc(
                        func=random_flip_func, shape=[batchSize], dtype=dt.UINT8, seed=seed_mediapipe)
                    self.random_flip = fn.RandomFlip(vertical=1)
                    flip = self.random_flip_input()
                    images = self.random_flip(images, flip)

                # elif isinstance(t, transforms.ToTensor):
                #    pass

                elif isinstance(t, transforms.ColorJitter):

                    if t.brightness != None:
                        brightness_params = {}
                        brightness_params['range'] = t.brightness
                        self.random_brightness = fn.MediaFunc(func=random_brigtness_func, shape=[
                                                              batchSize], priv_params=brightness_params, dtype=dt.FLOAT32, seed=seed_mediapipe)
                        self.brightness = fn.Brightness(brightness_scale=1)
                        brightness = self.random_brightness()
                        images = self.brightness(images, brightness)
                    if t.contrast != None:
                        contrast_params = {}
                        contrast_params['range'] = t.contrast
                        self.random_contrast = fn.MediaFunc(func=random_contrast_func, shape=[
                                                            batchSize], priv_params=contrast_params, dtype=dt.FLOAT32, seed=seed_mediapipe)
                        self.contrast = fn.Contrast(contrast_scale=1)
                        contrast = self.random_contrast()
                        images = self.contrast(images, contrast)
                    if t.saturation != None:
                        saturation_params = {}
                        saturation_params['range'] = t.saturation
                        self.random_saturation = fn.MediaFunc(func=random_saturation_func, shape=[
                                                              batchSize], priv_params=saturation_params, dtype=dt.FLOAT32, seed=seed_mediapipe)
                        self.saturation = fn.Saturation(saturation_level=1)
                        saturation = self.random_saturation()
                        images = self.saturation(images, saturation)
                    if t.hue != None:
                        hue_params = {}
                        hue_params['range'] = t.hue
                        self.random_hue = fn.MediaFunc(func=random_hue_func, shape=[
                                                       batchSize], priv_params=hue_params, dtype=dt.FLOAT32, seed=seed_mediapipe)
                        self.hue = fn.Hue(degree=0)
                        hue = self.random_hue()
                        images = self.hue(images, hue)

                else:
                    assert False, "Error: Unsupported transform" + str(type(t))
            transform_count += 1

        if (self.cmn_transform_index != None) or (self.ssd_train == True):
            # transforms.Normalize
            if (self.cmn_transform_index != None):
                t = self.transforms[self.cmn_transform_index]
            else:
                t = self.additional[Additional_Transform_Type.normalize]
            #normalize_mean = np.array([(0.485 * 255), (0.456 * 255), (0.406 * 255)], dtype=np.float32)
            #normalize_std = np.array([1 / (0.229 * 255), 1 / (0.224 * 255), 1 / (0.225 * 255)], dtype=np.float32)
            normalize_mean = np.array(
                [t.mean[0] * 255, t.mean[1] * 255, t.mean[2] * 255], dtype=np.float32)
            normalize_std = np.array(
                [1 / (t.std[0] * 255), 1 / (t.std[1] * 255), 1 / (t.std[2] * 255)], dtype=np.float32)
            normalize_scale = 0.03125

            # Define Constant tensors
            self.norm_mean = fn.MediaConst(data=normalize_mean, shape=[
                1, 1, 3], dtype=dt.FLOAT32)
            self.norm_std = fn.MediaConst(data=normalize_std, shape=[
                1, 1, 3], dtype=dt.FLOAT32)
            if crop_cmn == True:
                if self.media_output_dtype == 'uint8':
                    self.cmn = fn.CropMirrorNorm(crop_w=width, crop_h=height, crop_pos_x=cmn_pos_offset,
                                                 crop_pos_y=cmn_pos_offset, crop_d=0, output_scale=normalize_scale, output_zerop=128, dtype=dt.UINT8)
                elif self.media_output_dtype == 'float32':
                    self.cmn = fn.CropMirrorNorm(crop_w=width, crop_h=height, crop_pos_x=cmn_pos_offset,
                                                 crop_pos_y=cmn_pos_offset, crop_d=0, dtype=dt.FLOAT32)
                else:
                    assert False, "Data type not supported by Normalize"
                crop_cmn = False
            else:
                if self.media_output_dtype == 'uint8':
                    self.cmn = fn.CropMirrorNorm(
                        crop_w=width, crop_h=height, crop_d=0, output_scale=normalize_scale, output_zerop=128, dtype=dt.UINT8)
                elif self.media_output_dtype == 'float32':
                    self.cmn = fn.CropMirrorNorm(
                        crop_w=width, crop_h=height, crop_d=0, dtype=dt.FLOAT32)
                else:
                    assert False, "Data type not supported by Normalize"
            mean = self.norm_mean()
            std = self.norm_std()
            images = self.cmn(images, mean, std)

        #self.transpose = fn.Transpose(permutation=[2, 0, 1, 3], tensorDim=4)
        #images = self.transpose(images)
        #self.shape = [self.batchsize, 3, height, width]

        # self.setOutputShape(batch_size=batchSize, channel=3,
        #                    height=height, width=width, layout=lt.NCHW)
        if self.model_ssd == False:
            return images, data
        else:
            return images, ids, sizes, boxes, labels, lengths, batch
