#!/bin/env python
import numpy as np
import time
import glob
from habana_frameworks.mediapipe.operators.cpu_nodes.cpu_nodes import media_function
from habana_frameworks.mediapipe import fn  # NOQA
from habana_frameworks.mediapipe.mediapipe import MediaPipe  # NOQA
from habana_frameworks.mediapipe.media_types import imgtype as it  # NOQA
from habana_frameworks.mediapipe.media_types import dtype as dt  # NOQA

flip_prob = 0.33
brightness_prob = 0.1
noise_prob = 0.1
rbc_oversampling = 0.4


def get_random_seed():
    return int(time.time_ns() % (2**31 - 1))


class UnetMediaPipe(MediaPipe):
    def __init__(self, device, queue_depth, batch_size, input_list, patch_size, seed, drop_remainder, num_slices, slice_index, num_threads=1, is_testing=False):
        super().__init__(device=device,
                         prefetch_depth=queue_depth,
                         batch_size=batch_size,
                         num_threads=num_threads,
                         pipe_name=self.__class__.__name__)
        if(seed == None):
            seed = int(time.time_ns() % (2**31 - 1))
        self.seed_mediapipe = seed
        print("media data loader seed : ", self.seed_mediapipe)
        self.batch_size = batch_size
        self.patch_size = patch_size.copy()
        self.pipe_drop_last = drop_remainder
        self.num_slices = num_slices
        self.slice_index = slice_index
        self.is_testing = is_testing

        self.pipe_reader_pad_remainder = False
        train_shuffle_across_dataset = False
        train_shuffle = True

        val_shuffle_across_dataset = False
        val_shuffle = False

        # reader
        if(self.is_testing == False):
            self.images = fn.ReadNumpyDatasetFromDir(num_outputs=1,
                                                     shuffle=train_shuffle,
                                                     shuffle_across_dataset=train_shuffle_across_dataset,
                                                     file_list=input_list[0],
                                                     dtype=dt.FLOAT32,
                                                     # dense=False,
                                                     seed=self.seed_mediapipe,
                                                     drop_remainder=self.pipe_drop_last,
                                                     pad_remainder=self.pipe_reader_pad_remainder,
                                                     num_slices=self.num_slices,
                                                     slice_index=self.slice_index,
                                                     device='cpu',
                                                     cache_files=True
                                                     )

            self.labels = fn.ReadNumpyDatasetFromDir(num_outputs=1,
                                                     shuffle=train_shuffle,
                                                     shuffle_across_dataset=train_shuffle_across_dataset,
                                                     file_list=input_list[1],
                                                     dtype=dt.UINT8,
                                                     # dense=False,
                                                     seed=self.seed_mediapipe,
                                                     drop_remainder=self.pipe_drop_last,
                                                     pad_remainder=self.pipe_reader_pad_remainder,
                                                     num_slices=self.num_slices,
                                                     slice_index=self.slice_index,
                                                     device='cpu',
                                                     cache_files=True
                                                     )

            # random biased crop

            self.rand_bias_crop = fn.RandomBiasedCrop(patch_size=patch_size,
                                                      over_sampling=rbc_oversampling,
                                                      seed=get_random_seed(),
                                                      cache_bboxes=True,
                                                      cache_bboxes_at_first_run=False,
                                                      dtype=[dt.FLOAT32, dt.UINT8, dt.INT32], device='cpu')  # this output dtype is mandatory until cpp is generic

            # random flip
            self.random_flip_prob = fn.Constant(
                constant=flip_prob, dtype=dt.FLOAT32, device='cpu')

            self.flip_h = fn.CoinFlip(seed=get_random_seed(), device='cpu')
            self.flip_v = fn.CoinFlip(seed=get_random_seed(), device='cpu')
            self.flip_d = fn.CoinFlip(seed=get_random_seed(), device='cpu')

            self.img_hflip = fn.RandomFlip(horizontal=1, device='cpu')
            self.lbl_hflip = fn.RandomFlip(horizontal=1, device='cpu')
            self.img_vflip = fn.RandomFlip(vertical=1, device='cpu')
            self.lbl_vflip = fn.RandomFlip(vertical=1, device='cpu')
            self.img_dflip = fn.RandomFlip(depthwise=1, device='cpu')
            self.lbl_dflip = fn.RandomFlip(depthwise=1, device='cpu')

            # brightness
            self.brightness_probability = fn.Constant(
                constant=brightness_prob, dtype=dt.FLOAT32, device='cpu')

            self.coin_flip_b = fn.CoinFlip(
                seed=get_random_seed(), device='cpu')

            self.random_b = fn.RandomUniform(seed=get_random_seed(),
                                             low=0.7,
                                             high=1.3,
                                             device='cpu')
            self.const_b = fn.Constant(
                constant=1.0, dtype=dt.FLOAT32, device='cpu')
            self.where_b = fn.Where(device='cpu')
            self.brightness_op = fn.Mult(device='cpu')

            # gaussian noise
            self.gnoise_probability = fn.Constant(
                constant=noise_prob, dtype=dt.FLOAT32, device='cpu')

            self.coin_flip_g = fn.CoinFlip(
                seed=get_random_seed(), device='cpu'
            )
            self.random_g = fn.RandomNormal(
                seed=get_random_seed(), mean=0, device='cpu')
            self.where_g = fn.Where(device='cpu')
            self.noise_op = fn.Add(device='cpu')
            self.const_std_dev_g = fn.Constant(
                constant=0.1, dtype=dt.FLOAT32, device='cpu')
            self.const_zero_g = fn.Constant(
                constant=0, dtype=dt.FLOAT32, device='cpu')
        else:
            self.val_images = fn.ReadNumpyDatasetFromDir(num_outputs=1,
                                                         shuffle=val_shuffle,
                                                         shuffle_across_dataset=val_shuffle_across_dataset,
                                                         file_list=input_list[0],
                                                         dtype=dt.FLOAT32,
                                                         # dense=False,
                                                         seed=self.seed_mediapipe,
                                                         drop_remainder=self.pipe_drop_last,
                                                         pad_remainder=self.pipe_reader_pad_remainder,
                                                         num_slices=self.num_slices,
                                                         slice_index=self.slice_index,
                                                         device='cpu',
                                                         cache_files=True
                                                         )

            self.val_labels = fn.ReadNumpyDatasetFromDir(num_outputs=1,
                                                         shuffle=val_shuffle,
                                                         shuffle_across_dataset=val_shuffle_across_dataset,
                                                         file_list=input_list[1],
                                                         dtype=dt.UINT8,
                                                         # dense=False,
                                                         seed=self.seed_mediapipe,
                                                         drop_remainder=self.pipe_drop_last,
                                                         pad_remainder=self.pipe_reader_pad_remainder,
                                                         num_slices=self.num_slices,
                                                         slice_index=self.slice_index,
                                                         device='cpu',
                                                         cache_files=True
                                                         )

    def definegraph(self):
        if(self.is_testing == False):
            img = self.images()
            lbl = self.labels()

            # biased crop
            img, lbl, coord = self.rand_bias_crop(img, lbl)

            # random flips
            rflip_prob = self.random_flip_prob()
            h_predicate = self.flip_h(rflip_prob)
            v_predicate = self.flip_v(rflip_prob)
            d_predicate = self.flip_d(rflip_prob)
            img = self.img_hflip(img, h_predicate)
            lbl = self.lbl_hflip(lbl, h_predicate)
            img = self.img_vflip(img, v_predicate)
            lbl = self.lbl_vflip(lbl, v_predicate)
            img = self.img_dflip(img, d_predicate)
            lbl = self.lbl_dflip(lbl, d_predicate)

            # brightness
            b_prob = self.brightness_probability()
            scale = self.random_b()
            b_predicate = self.coin_flip_b(b_prob)
            scale_def = self.const_b()  # 1.0
            scale = self.where_b(b_predicate, scale, scale_def)
            #scale = self.where_b(b_predicate, scale_def, scale)
            img = self.brightness_op(img, scale)

            # gaussian noise

            gn_prob = self.gnoise_probability()
            g_predicate = self.coin_flip_g(gn_prob)
            std_dev_in = self.const_std_dev_g()  # 0.1
            std_dev_def = self.const_zero_g()  # 0
            std_dev = self.where_g(g_predicate, std_dev_in, std_dev_def)
            noiseVal = self.random_g(std_dev, img)
            img = self.noise_op(img, noiseVal)

            return img, lbl
        else:
            img = self.val_images()
            lbl = self.val_labels()
            return img, lbl
