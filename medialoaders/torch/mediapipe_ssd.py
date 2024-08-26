import numpy as np
import time

from habana_frameworks.mediapipe import fn  # NOQA
from habana_frameworks.mediapipe.mediapipe import MediaPipe  # NOQA
from habana_frameworks.mediapipe.media_types import imgtype as it  # NOQA
from habana_frameworks.mediapipe.media_types import dtype as dt  # NOQA
from habana_frameworks.mediapipe.media_types import ftype as ft  # NOQA
from habana_frameworks.mediapipe.media_types import decoderStage as ds  # NOQA
from habana_frameworks.mediapipe.media_types import lastBatchStrategy as lbs #NOQA

# bbox flip prob
flip_prob = 0.5

# brightness
brt_factor_min = 0.875  # max(0, 1-0.125)
brt_factor_max = 1.125  # 1+0.125

# contrast
con_scale_min = 0.5  # max(0, 0-0.5)
con_scale_max = 1.5  # 1+0.5

# saturation
sat_level_min = 0.5  # max(0,1-0.5)
sat_level_max = 1.5  # 1+0.5

# hue
hue_degree_min = -18.0  # -0.05 * 360
hue_degree_max = 18.0  # 0.05 * 360


def get_random_seed():
    return int(time.time_ns() % (2**31 - 1))


class SSDMediaPipe(MediaPipe):
    def __init__(self, a_device=None, a_is_train=True, a_root=None, a_annotation_file=None, a_width=300, a_height=300, a_batch_size=1, a_shuffle=False, a_drop_last=True, a_prefetch_count=1, a_num_instances=1, a_instance_id=0, a_num_threads=1):

        super(
            SSDMediaPipe,
            self).__init__(
            a_device,
            a_prefetch_count,
            a_batch_size,
            a_num_threads,
            self.__class__.__name__)

        print("SSDMediaPipe device {} is_train {} root {} annotation_file {} width {} height {} batch_size {} shuffle {} drop_last {} prefetch_count {} num_instances {} instance_id {} num_threads {}".format(
            a_device, a_is_train, a_root, a_annotation_file, a_width, a_height, a_batch_size, a_shuffle, a_drop_last, a_prefetch_count, a_num_instances, a_instance_id, a_num_threads))

        self.dataset_path = a_root
        self.annotation_file = a_annotation_file
        self.batch_size = a_batch_size

        self.num_slices = a_num_instances
        self.slice_index = a_instance_id

        self.is_train = a_is_train
        self.pipe_seed = get_random_seed()

        self.train_shuffle = a_shuffle
        self.train_drop_remainder = False
        self.train_partial_batch = True

        self.val_shuffle = a_shuffle
        self.val_drop_remainder = False
        self.val_partial_batch = True

        self.height = a_height
        self.width = a_width

        if(self.is_train):
            self.input = fn.CocoReader(root=self.dataset_path,
                                       annfile=self.annotation_file,
                                       seed=self.pipe_seed,
                                       shuffle=self.train_shuffle,
                                       drop_remainder=self.train_drop_remainder,
                                       num_slices=self.num_slices,
                                       slice_index=self.slice_index,
                                       last_batch_strategy = lbs.PARTIAL,
                                       device='cpu')

            self.reshape_ids = fn.Reshape(size=[self.batch_size],
                                          tensorDim=1,
                                          layout='',
                                          dtype=dt.INT32, device='hpu')  # [batch_size]

            self.ssd_crop_win_gen = fn.SSDCropWindowGen(
                num_iterations=1, seed=get_random_seed(), device='cpu')

            self.bbox_flip_prob = fn.Constant(
                constant=flip_prob, dtype=dt.FLOAT32, device='cpu')

            self.is_bbox_flip = fn.CoinFlip(
                seed=get_random_seed(), dtype=dt.INT8, device='cpu')

            self.ssd_bbox_flip = fn.SSDBBoxFlip(device='cpu')

            self.ssd_encode = fn.SSDEncode(device='cpu')
            # Decodee
            def_output_image_size = [self.width, self.height]

            self.decode = fn.ImageDecoder(device="hpu",
                                          output_format=it.RGB_P,
                                          resize=def_output_image_size,
                                          resampling_mode=ft.BI_LINEAR
                                          )
            # iamge flip - Horizontal
            self.reshape_is_flip = fn.Reshape(size=[self.batch_size],
                                              tensorDim=1,
                                              layout='',
                                              dtype=dt.UINT8, device='hpu')
            self.random_flip = fn.RandomFlip(horizontal=1, device='hpu')

            # brightness
            self.random_b_scale = fn.RandomUniform(seed=get_random_seed(
            ), low=brt_factor_min, high=brt_factor_max, dtype=dt.FLOAT32, device='cpu')

            self.reshape_b_scale = fn.Reshape(size=[self.batch_size],
                                              tensorDim=1,
                                              layout='',
                                              dtype=dt.FLOAT32, device='hpu')  # [batch_size]

            self.brightness = fn.Brightness(brightness_scale=1, device='hpu')

            # Contrast
            self.random_c_scale = fn.RandomUniform(seed=get_random_seed(
            ), low=con_scale_min, high=con_scale_max, dtype=dt.FLOAT32, device='cpu')

            self.reshape_c_scale = fn.Reshape(size=[self.batch_size],
                                              tensorDim=1,
                                              layout='',
                                              dtype=dt.FLOAT32, device='hpu')  # [batch_size]

            self.contrast = fn.Contrast(contrast_scale=1, device='hpu')

            # Saturation
            self.random_s_level = fn.RandomUniform(seed=get_random_seed(
            ), low=sat_level_min, high=sat_level_max, dtype=dt.FLOAT32, device='cpu')

            self.reshape_s_level = fn.Reshape(size=[self.batch_size],
                                              tensorDim=1,
                                              layout='',
                                              dtype=dt.FLOAT32, device='hpu')  # [batch_size]

            self.saturation = fn.Saturation(saturation_level=1, device='hpu')

            # hue
            self.random_h_degree = fn.RandomUniform(
                seed=get_random_seed(), low=hue_degree_min, high=hue_degree_max, dtype=dt.FLOAT32, device='cpu')
            self.modulo_degree = fn.Modulo(divisor=360, device='cpu')
            self.reshape_h_degree = fn.Reshape(size=[self.batch_size],
                                               tensorDim=1,
                                               layout='',
                                               dtype=dt.FLOAT32, device='hpu')  # [batch_size]
            self.hue = fn.Hue(degree=0, device='hpu')

            # CMN
            # model_garden/PyTorch/computer_vision/detection/mlcommons/SSD/ssd/utils.py
            #normalization_mean = [0.485, 0.456, 0.406]
            #normalization_std = [0.229, 0.224, 0.225]
            normalize_mean = np.array(
                [(0.485 * 255), (0.456 * 255), (0.406 * 255)], dtype=np.float32)
            normalize_std = np.array(
                [1 / (0.229 * 255), 1 / (0.224 * 255), 1 / (0.225 * 255)], dtype=np.float32)

            # Define Constant tensors
            self.norm_mean = fn.MediaConst(data=normalize_mean, shape=[
                1, 1, 3], dtype=dt.FLOAT32, device='cpu')
            self.norm_std = fn.MediaConst(data=normalize_std, shape=[
                1, 1, 3], dtype=dt.FLOAT32, device='cpu')

            self.cmn = fn.CropMirrorNorm(
                crop_w=self.width, crop_h=self.height, crop_d=0, dtype=dt.FLOAT32, device='hpu')
        else:
            # val data loader
            self.input = fn.CocoReader(root=self.dataset_path,
                                       annfile=self.annotation_file,
                                       seed=self.pipe_seed,
                                       shuffle=self.val_shuffle,
                                       drop_remainder=self.val_drop_remainder,
                                       num_slices=self.num_slices,
                                       slice_index=self.slice_index,
                                       last_batch_strategy = lbs.PARTIAL,
                                       device='cpu')

            self.reshape_ids = fn.Reshape(size=[self.batch_size],
                                          tensorDim=1,
                                          layout='',
                                          dtype=dt.INT32, device='hpu')  # [batch_size]

            # Decode
            def_output_image_size = [self.width, self.height]

            self.decode = fn.ImageDecoder(device="hpu",
                                          output_format=it.RGB_P,
                                          resize=def_output_image_size,
                                          resampling_mode=ft.BI_LINEAR
                                          )
            #normalization_mean = [0.485, 0.456, 0.406]
            #normalization_std = [0.229, 0.224, 0.225]
            normalize_mean = np.array(
                [(0.485 * 255), (0.456 * 255), (0.406 * 255)], dtype=np.float32)
            normalize_std = np.array(
                [1 / (0.229 * 255), 1 / (0.224 * 255), 1 / (0.225 * 255)], dtype=np.float32)

            # Define Constant tensors
            self.norm_mean = fn.MediaConst(data=normalize_mean, shape=[
                1, 1, 3], dtype=dt.FLOAT32, device='cpu')
            self.norm_std = fn.MediaConst(data=normalize_std, shape=[
                1, 1, 3], dtype=dt.FLOAT32, device='cpu')

            self.cmn = fn.CropMirrorNorm(
                crop_w=self.width, crop_h=self.height, crop_d=0, dtype=dt.FLOAT32, device='hpu')

    def definegraph(self):
        if(self.is_train):
            # Train pipe
            jpegs, ids, sizes, boxes, labels, lengths, batch = self.input()

            # ssd crop window generation
            sizes, boxes, labels, lengths, windows = self.ssd_crop_win_gen(
                sizes, boxes, labels, lengths)

            # ssd Bounding box flip
            bb_flip_prob = self.bbox_flip_prob()
            is_Flip = self.is_bbox_flip(bb_flip_prob)
            boxes = self.ssd_bbox_flip(is_Flip, boxes, lengths)

            # ssd encode
            boxes, labels = self.ssd_encode(boxes, labels, lengths)
            images = self.decode(jpegs, windows)

            # image flip
            is_Flip = self.reshape_is_flip(is_Flip)
            images = self.random_flip(images, is_Flip)

            # brightness
            scale_b = self.random_b_scale()
            scale_b = self.reshape_b_scale(scale_b)
            images = self.brightness(images, scale_b)

            # contrast
            scale_c = self.random_c_scale()
            scale_c = self.reshape_c_scale(scale_c)
            images = self.contrast(images, scale_c)

            # saturation
            s_level = self.random_s_level()
            s_level = self.reshape_s_level(s_level)
            images = self.saturation(images, s_level)

            # hue
            h_degree = self.random_h_degree()
            h_degree = self.modulo_degree(h_degree)
            h_degree = self.reshape_h_degree(h_degree)
            images = self.hue(images, h_degree)

            # cmn
            mean = self.norm_mean()
            std = self.norm_std()
            images = self.cmn(images, mean, std)

            # create hpu tensors
            ids = self.reshape_ids(ids)
            sizes.as_hpu()
            boxes.as_hpu()
            labels.as_hpu()
            lengths.as_hpu()

            return images, ids, sizes, boxes, labels, lengths, batch
        else:
            # Val pipe
            jpegs, ids, sizes, boxes, labels, lengths, batch = self.input()
            images = self.decode(jpegs)
            # cmn
            mean = self.norm_mean()
            std = self.norm_std()
            images = self.cmn(images, mean, std)
            ids = self.reshape_ids(ids)
            sizes.as_hpu()
            boxes.as_hpu()
            labels.as_hpu()
            lengths.as_hpu()
            return images, ids, sizes, boxes, labels, lengths, batch
