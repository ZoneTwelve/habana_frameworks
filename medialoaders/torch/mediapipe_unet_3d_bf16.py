import numpy as np
import time
import math
from habana_frameworks.mediapipe.operators.cpu_nodes.cpu_nodes import media_function
from habana_frameworks.mediapipe import fn  # NOQA
from habana_frameworks.mediapipe.mediapipe import MediaPipe  # NOQA
from habana_frameworks.mediapipe.media_types import dtype as dt  # NOQA

# augmentation probability
g_rand_aug_prob = 0.15

# zoom
g_crop_min = 0.7
g_crop_max = 1.0

# flip
flip_priv_params = {
    'prob': 0.33
}

# noise
g_min_std_dev = 0.0
g_max_std_dev = 0.33

# gaussian blur
g_max_sigma = 1.5
g_min_sigma = 0.5

# brightness
g_brt_factor_min = 0.7
g_brt_factor_max = 1.3

# contrast
g_scale_min = 0.65
g_scale_max = 1.5


class Unet3dMediaPipe(MediaPipe):

    """
    Class defining unet media pipe.

    """

    def __init__(self, a_device=None, a_batch_size=1, a_prefetch_count=1, a_num_instances=1, a_instance_id=0, a_pipeline=None, a_enable_zoom=False, a_num_threads=1, **kwargs):
        """
        Constructor method.

        :params a_device: media device to run mediapipe on. <hpu/hpu:0>
        :params a_batch_size: mediapipe output batch size.
        :params a_prefetch_count: queue depth for media processing.
        :params a_num_instances: number of devices.
        :params a_instance_id: instance id of current device.
        :params a_pipeline: mediapipe to be created for Unet.<TrainPipeline/BenchmarkPipeline_Train>
        :params a_enable_zoom: whether zoom has to be enabled for TrainPipeline
        :params **kwargs: dictionary of params for Unet3dMediaPipe
        """

        self.batch_size = a_batch_size
        prefetch_count = a_prefetch_count
        self.num_instances = a_num_instances
        self.instance_id = a_instance_id
        self.pipeName = a_pipeline
        self.enable_zoom = a_enable_zoom
        self.dim = kwargs["dim"]
        # seed = kwargs["seed"]
        seed_mediapipe = kwargs.get("seed", None)
        self.patch_size = kwargs["patch_size"]
        self.oversampling = kwargs["oversampling"]  # 0.33
        # self.patch_size = patch_size.copy()
        pipe_drop_last = False
        pipe_reader_pad_remainder = True
        if(self.pipeName == "BenchmarkPipeline_Train"):
            pipe_num_workers = 2
        else:
            pipe_num_workers = min(self.batch_size, 4)
        train_shuffle = True
        train_shuffle_across_dataset = True
        benchmark_shuffle = False
        benchmark_shuffle_across_dataset = False
        val_shuffle = False
        val_shuffle_across_dataset = False
        test_shuffle_across_dataset = False
        test_shuffle = False
        image_num_channel = 4
        label_num_channel = 1

        class_pipename = "{}:{}".format(self.__class__.__name__, self.pipeName)
        class_pipename = str(class_pipename)
        super().__init__(device=a_device,
                         prefetch_depth=prefetch_count,
                         batch_size=self.batch_size,
                         num_threads=a_num_threads,
                         pipe_name=class_pipename)

        if (self.pipeName != "TrainPipeline") and (self.pipeName != "BenchmarkPipeline_Train") and (self.pipeName != "EvalPipeline") and (self.pipeName != "TestPipeline"):
            raise ValueError(
                "Unet3dMediaPipe: pipe {} not supported!".format(self.pipeName))

        if(self.pipeName == "TestPipeline"):
            input_list = [kwargs["imgs"], kwargs["meta"]]
        else:
            input_list = [kwargs["imgs"], kwargs["lbls"]]

        if (self.pipeName == "TrainPipeline"):
            self.augment = kwargs["augment"]
            set_aug_seed = kwargs["set_aug_seed"]

        # if set_aug_seed is False, generate new seed for augmentation
        # if set_aug_seed is True and seed_mediapipe is available, use it for augmentation
        if(seed_mediapipe == None):
            if (self.num_instances > 1) and (self.pipeName == "TrainPipeline") and (train_shuffle_across_dataset == True):
                raise ValueError(
                    "Unet3dMediaPipe: num_instances > 1, seed not provided!")
            else:
                print("Warning: Unet3dMediaPipe seed not provided, generating seed")

            seed_mediapipe = int(time.time_ns() % (2**31 - 1))
            if (self.pipeName == "TrainPipeline"):
                seed_augment = seed_mediapipe
        else:
            if self.pipeName == "TrainPipeline":
                if set_aug_seed == False:
                    seed_augment = int(time.time_ns() % (2**31 - 1))
                else:
                    seed_augment = seed_mediapipe + self.instance_id

        print("Unet3dMediaPipe BFLOAT16 pipeline {} batch_size {} dim {} oversampling {} prefetch_depth {}".format(
            self.pipeName, self.batch_size, self.dim, self.oversampling, prefetch_count))

        if self.pipeName == "TrainPipeline":
            print("Unet3dMediaPipe seed {} augment {} set_aug_seed {} augment_seed {} num_instances {} instance_id {}".format(
                seed_mediapipe, self.augment, set_aug_seed, seed_augment, self.num_instances, self.instance_id))

            if self.augment == False:
                print("Warning: Unet3dMediaPipe augmentation turned off")

            if self.enable_zoom == True:
                print("Unet3dMediaPipe zoom enabled")

        else:
            print("Unet3dMediaPipe seed {} num_instances {} instance_id {}".format(
                seed_mediapipe, self.num_instances, self.instance_id))

        if (self.pipeName == "TrainPipeline") or (self.pipeName == "BenchmarkPipeline_Train"):
            if self.dim == 2:
                self.patch_size = self.patch_size[::-1]
                # self.patch_size = [kwargs["batch_size_2d"]] + self.patch_size
                self.patch_size = self.patch_size + [kwargs["batch_size_2d"]]
                print("Unet3dMediaPipe patch size ", self.patch_size[::-1])
            else:
                self.patch_size = self.patch_size[::-1]
                print("Unet3dMediaPipe patch size ", self.patch_size[::-1])

        if (self.pipeName == "TrainPipeline"):
            # ------------reader------------------------------
            self.inputxy = fn.ReadNumpyDatasetFromDir(num_outputs=2,
                                                      shuffle=train_shuffle,
                                                      shuffle_across_dataset=train_shuffle_across_dataset,
                                                      file_list=input_list,
                                                      dtype=[
                                                          dt.FLOAT32, dt.UINT8],
                                                      dense=False,
                                                      seed=seed_mediapipe,
                                                      num_readers=pipe_num_workers,
                                                      drop_remainder=pipe_drop_last,
                                                      pad_remainder=pipe_reader_pad_remainder,
                                                      num_slices=self.num_instances,
                                                      slice_index=self.instance_id)

            # ------------random biased crop------------------------------

            self.rand_bias_crop = fn.RandomBiasedCrop(patch_size=self.patch_size,
                                                      over_sampling=self.oversampling,
                                                      seed=seed_augment,
                                                      num_channels=image_num_channel,
                                                      num_workers=pipe_num_workers)

            if self.enable_zoom == True:

                priv_params = {}
                priv_params['prob'] = g_rand_aug_prob
                priv_params['crop_min'] = g_crop_min
                priv_params['crop_max'] = g_crop_max
                priv_params['patch_size'] = self.patch_size

                self.crop_size = fn.MediaFunc(func=random_zoom_func,
                                              shape=[3, self.batch_size],
                                              dtype=dt.UINT32,
                                              seed=seed_augment,
                                              priv_params=priv_params)

                self.zoom = fn.Zoom(patch_size=self.patch_size,
                                    num_channels=image_num_channel)

            # ------------Reshape(for flip)------------------------------
            self.cast_bf16 = fn.Cast(dtype=dt.BFLOAT16)
            # Reshape for flip [ W,H,D,C,N] -> [W,H,D*C,N]
            shape_patch = self.patch_size.copy()
            shape = []
            shape.append(shape_patch[0])
            shape.append(shape_patch[1])
            shape.append(shape_patch[2]*image_num_channel)
            shape.append(self.batch_size)
            self.img_reshape_hflip = fn.Reshape(size=shape,
                                                tensorDim=len(shape),
                                                layout='',
                                                dtype=dt.BFLOAT16)
            shape_patch = self.patch_size.copy()
            shape = []
            shape.append(shape_patch[0])
            shape.append(shape_patch[1])
            shape.append(shape_patch[2]*label_num_channel)
            shape.append(self.batch_size)

            self.lbl_reshape_hflip = fn.Reshape(size=shape,
                                                tensorDim=len(shape),
                                                layout='',
                                                dtype=dt.UINT8)
            # ------------H Flip------------------------------
            # random horizontal flip node
            self.is_hflip = fn.MediaFunc(func=random_flip_func,
                                         shape=[self.batch_size],
                                         dtype=dt.UINT8,
                                         seed=seed_augment,
                                         priv_params=flip_priv_params)
            self.img_hflip = fn.RandomFlip(horizontal=1, dtype=dt.BFLOAT16)
            self.lbl_hflip = fn.RandomFlip(horizontal=1, dtype=dt.UINT8)

            # ------------V Flip------------------------------
            # random vertical flip node
            self.is_vflip = fn.MediaFunc(func=random_flip_func,
                                         shape=[self.batch_size],
                                         seed=seed_augment,
                                         dtype=dt.UINT8,
                                         priv_params=flip_priv_params)
            self.img_vflip = fn.RandomFlip(vertical=1, dtype=dt.BFLOAT16)
            self.lbl_vflip = fn.RandomFlip(vertical=1, dtype=dt.UINT8)

            if self.dim == 3:
                # ------------D Flip------------------------------
                # Reshape for dflip [W,H,D*C,N] -> [W*H,D,C,N]
                shape_patch = self.patch_size.copy()
                shape = []
                shape.append(shape_patch[0] * shape_patch[1])
                shape.append(shape_patch[2])
                shape.append(image_num_channel)
                shape.append(self.batch_size)
                self.img_reshape_dflip = fn.Reshape(size=shape,
                                                    tensorDim=len(shape),
                                                    layout='',
                                                    dtype=dt.BFLOAT16)

                shape_patch = self.patch_size.copy()
                shape = []
                shape.append(shape_patch[0] * shape_patch[1])
                shape.append(shape_patch[2])
                shape.append(label_num_channel)
                shape.append(self.batch_size)
                self.lbl_reshape_dflip = fn.Reshape(size=shape,
                                                    tensorDim=len(shape),
                                                    layout='',
                                                    dtype=dt.UINT8)

                self.is_dflip = fn.MediaFunc(func=random_flip_func,
                                             shape=[self.batch_size],
                                             seed=seed_augment,
                                             dtype=dt.UINT8,
                                             priv_params=flip_priv_params)
                self.img_dflip = fn.RandomFlip(vertical=1, dtype=dt.BFLOAT16)
                self.lbl_dflip = fn.RandomFlip(vertical=1, dtype=dt.UINT8)

                # Reshape [W*H,D,C,N] -> [W,H,D*C,N]
                shape_patch = self.patch_size.copy()
                shape = []
                shape.append(shape_patch[0])
                shape.append(shape_patch[1])
                shape.append(shape_patch[2]*image_num_channel)
                shape.append(self.batch_size)
                self.img_reshape_noise = fn.Reshape(size=shape,
                                                    tensorDim=len(shape),
                                                    layout='',
                                                    dtype=dt.BFLOAT16)

            # ------------Gaussian noise------------------------------

            self.seed = fn.MediaFunc(func=random_seed_func,
                                     shape=[1],
                                     dtype=dt.UINT32,
                                     seed=seed_augment)
            priv_params = {}
            # priv_params['batch_size'] = self.batch_size
            priv_params['min_std_dev'] = g_min_std_dev
            priv_params['max_std_dev'] = g_max_std_dev
            priv_params['prob'] = g_rand_aug_prob
            self.stddev = fn.MediaFunc(func=random_stddev_func,
                                       shape=[self.batch_size],
                                       dtype=dt.FLOAT32,
                                       seed=seed_augment,
                                       priv_params=priv_params)

            shape_patch = self.patch_size.copy()
            shape = []
            shape.append(shape_patch[0])
            shape.append(shape_patch[1])
            shape.append(shape_patch[2]*image_num_channel)
            shape.append(self.batch_size)
            self.rnd_normal = fn.RandomNormal(mean=0.0,
                                              stddev=0.1,
                                              dtype=dt.BFLOAT16,
                                              dims=len(shape),
                                              shape=shape)
            self.add = fn.Add(dtype=dt.BFLOAT16)

            # Gaussian Blur
            priv_params_sigma = {}
            priv_params_sigma['batch_size'] = self.batch_size
            priv_params_sigma['min_sigma'] = g_min_sigma
            priv_params_sigma['max_sigma'] = g_max_sigma

            priv_params_apply = {}
            priv_params_apply['batch_size'] = self.batch_size
            priv_params_apply['prob'] = g_rand_aug_prob

            self.gaussian_kernel_sigma = fn.MediaFunc(func=gaussian_kernel_sigma, shape=([
                self.batch_size]), seed=seed_augment, dtype=dt.FLOAT32, priv_params=priv_params_sigma)

            self.gaussian_kernel_apply_blur = fn.MediaFunc(func=gaussian_kernel_apply_blur, shape=([
                self.batch_size]), seed=seed_augment, dtype=dt.FLOAT32, priv_params=priv_params_apply)
            shape = self.patch_size.copy()  # [W,H,D]
            shape.append(image_num_channel)
            shape.append(self.batch_size)
            self.img_reshape_transpose_pre_blur = fn.Reshape(size=shape,
                                                             tensorDim=len(
                                                                 shape),
                                                             layout='',
                                                             dtype=dt.BFLOAT16,
                                                             device='hpu')  # img [W,H,D,C,N]
            self.gaussian_blur = fn.GaussianBlur(
                max_sigma=g_max_sigma, min_sigma=g_min_sigma, shape=shape, dtype=dt.BFLOAT16)
            # ------------Brightness------------------------------
            # Reshape [W,H,D,C,N] ->  [W,H,D*C,N]
            shape_patch = self.patch_size.copy()
            shape = []
            shape.append(shape_patch[0])
            shape.append(shape_patch[1])
            shape.append(shape_patch[2]*image_num_channel)
            shape.append(self.batch_size)
            self.img_reshape_brightness = fn.Reshape(size=shape,
                                                     tensorDim=len(shape),
                                                     layout='',
                                                     dtype=dt.BFLOAT16)

            priv_params = {}
            priv_params['factor_min'] = g_brt_factor_min
            priv_params['factor_max'] = g_brt_factor_max
            priv_params['prob'] = g_rand_aug_prob
            # priv_params['batch_size'] = self.batch_size

            self.brt_in = fn.MediaFunc(func=brightness_func,
                                       seed=seed_augment,
                                       shape=[self.batch_size],
                                       dtype=dt.FLOAT32,
                                       priv_params=priv_params)
            # priv_params=brt_priv_params)
            self.brightness = fn.Brightness(dtype=dt.BFLOAT16)

            # ------------Contrast------------------------------
            shape = self.patch_size.copy()
            shape.append(image_num_channel)
            shape.append(self.batch_size)
            self.img_reshape_contrast_input = fn.Reshape(size=shape,
                                                         tensorDim=len(shape),
                                                         layout='',
                                                         dtype=dt.BFLOAT16)
            priv_params = {}
            priv_params['scale_min'] = g_scale_min
            priv_params['scale_max'] = g_scale_max
            priv_params['prob'] = g_rand_aug_prob
            # priv_params['batch_size'] = self.batch_size

            self.scale = fn.MediaFunc(func=random_scale_func,
                                      shape=[1, self.batch_size],
                                      dtype=dt.FLOAT32,
                                      seed=seed_augment,
                                      priv_params=priv_params)
            self.min = fn.ReduceMin(
                reductionDimension=[3, 2, 1, 0], dtype=dt.BFLOAT16)
            self.max = fn.ReduceMax(
                reductionDimension=[3, 2, 1, 0], dtype=dt.BFLOAT16)
            shape = []
            shape.append(
                self.patch_size[0]*self.patch_size[1]*self.patch_size[2]*image_num_channel)
            shape.append(self.batch_size)
            self.img_reshape_before_mul = fn.Reshape(size=shape,
                                                        tensorDim=2,
                                                        layout='',
                                                        dtype=dt.BFLOAT16)
            self.mul = fn.Mult(dtype=dt.BFLOAT16)

            shape = self.patch_size.copy()  # [WHD]
            shape.append(image_num_channel)  # [WHDC]
            shape.append(self.batch_size)  # [WHDN]
            self.img_reshape_after_mul = fn.Reshape(size=shape,
                                                         tensorDim=len(shape),
                                                         layout='',
                                                         dtype=dt.BFLOAT16)  # img [W,H,D,C,N]

            self.clamp = fn.Clamp(dtype=dt.BFLOAT16)

            # ------------Reshape------------------------------
            #[W,H,D,C,N]
            shape = self.patch_size.copy()
            shape.append(image_num_channel)
            shape.append(self.batch_size)
            self.img_reshape_output = fn.Reshape(size=shape,
                                                 tensorDim=len(shape),
                                                 layout='',
                                                 dtype=dt.BFLOAT16)

            shape = self.patch_size.copy()
            shape.append(label_num_channel)
            shape.append(self.batch_size)
            self.lbl_reshape_output = fn.Reshape(size=shape,
                                                 tensorDim=len(shape),
                                                 layout='',
                                                 dtype=dt.UINT8)

            if self.dim == 2:
                # ------------Transpose------------------------------
                # [W,H,D,C,N] -> [W,H,C,D,N]
                self.img_transpose = fn.Transpose(permutation=[0, 1, 3, 2, 4],
                                                  tensorDim=5,
                                                  dtype=dt.BFLOAT16)
                self.lbl_transpose = fn.Transpose(permutation=[0, 1, 3, 2, 4],
                                                  tensorDim=5,
                                                  dtype=dt.UINT8)

        elif (self.pipeName == "BenchmarkPipeline_Train"):

            self.inputxy = fn.ReadNumpyDatasetFromDir(num_outputs=2,
                                                      shuffle=benchmark_shuffle,
                                                      shuffle_across_dataset=benchmark_shuffle_across_dataset,
                                                      file_list=input_list,
                                                      dtype=[
                                                          dt.FLOAT32, dt.UINT8],
                                                      dense=False,
                                                      seed=seed_mediapipe,
                                                      num_readers=pipe_num_workers,
                                                      drop_remainder=pipe_drop_last,
                                                      pad_remainder=pipe_reader_pad_remainder,
                                                      num_slices=self.num_instances,
                                                      slice_index=self.instance_id)

            self.crop_img = fn.BasicCrop(patch_size=self.patch_size,
                                         dtype=dt.FLOAT32,
                                         num_channels=image_num_channel,
                                         center_crop=True)
            self.crop_lbl = fn.BasicCrop(patch_size=self.patch_size,
                                         dtype=dt.UINT8,
                                         num_channels=label_num_channel,
                                         center_crop=True)
            self.cast_bf16 = fn.Cast(dtype=dt.BFLOAT16)
            shape = self.patch_size.copy()
            shape.append(image_num_channel)
            shape.append(self.batch_size)
            self.img_reshape_output = fn.Reshape(size=shape,
                                                 tensorDim=len(shape),
                                                 layout='',
                                                 dtype=dt.BFLOAT16)
            shape = self.patch_size.copy()
            shape.append(label_num_channel)
            shape.append(self.batch_size)
            self.lbl_reshape_output = fn.Reshape(size=shape,
                                                 tensorDim=len(shape),
                                                 layout='',
                                                 dtype=dt.UINT8)
            if self.dim == 2:
                # [W,H,D,C,N] -> [W,H,C,D,N]
                self.img_transpose = fn.Transpose(permutation=[0, 1, 3, 2, 4],
                                                  tensorDim=5,
                                                  dtype=dt.BFLOAT16)
                self.lbl_transpose = fn.Transpose(permutation=[0, 1, 3, 2, 4],
                                                  tensorDim=5,
                                                  dtype=dt.UINT8)

        elif(self.pipeName == "EvalPipeline"):
            self.inputX = fn.ReadNumpyDatasetFromDir(num_outputs=1,
                                                     shuffle_across_dataset=val_shuffle_across_dataset,
                                                     shuffle=val_shuffle,
                                                     file_list=input_list[0],
                                                     dtype=dt.FLOAT32,
                                                     # dense=False,
                                                     seed=seed_mediapipe,
                                                     drop_remainder=pipe_drop_last,
                                                     pad_remainder=pipe_reader_pad_remainder,
                                                     num_slices=self.num_instances,
                                                     slice_index=self.instance_id,
                                                     device='cpu'
                                                     )
            self.inputY = fn.ReadNumpyDatasetFromDir(num_outputs=1,
                                                     shuffle_across_dataset=val_shuffle_across_dataset,
                                                     shuffle=val_shuffle,
                                                     file_list=input_list[1],
                                                     dtype=dt.UINT8,
                                                     # dense=False,
                                                     seed=seed_mediapipe,
                                                     drop_remainder=pipe_drop_last,
                                                     pad_remainder=pipe_reader_pad_remainder,
                                                     num_slices=self.num_instances,
                                                     slice_index=self.instance_id,
                                                     device='cpu'
                                                     )
        elif(self.pipeName == "TestPipeline"):
            self.inputX = fn.ReadNumpyDatasetFromDir(num_outputs=1,
                                                     shuffle_across_dataset=test_shuffle_across_dataset,
                                                     shuffle=test_shuffle,
                                                     file_list=input_list[0],
                                                     dtype=dt.FLOAT32,
                                                     # dense=False,
                                                     seed=seed_mediapipe,
                                                     drop_remainder=pipe_drop_last,
                                                     pad_remainder=pipe_reader_pad_remainder,
                                                     num_slices=self.num_instances,
                                                     slice_index=self.instance_id,
                                                     device='cpu'
                                                     )
            self.inputMeta = fn.ReadNumpyDatasetFromDir(num_outputs=1,
                                                        shuffle_across_dataset=test_shuffle_across_dataset,
                                                        shuffle=test_shuffle,
                                                        file_list=input_list[1],
                                                        dtype=dt.INT64,
                                                        # dense=False,
                                                        seed=seed_mediapipe,
                                                        drop_remainder=pipe_drop_last,
                                                        pad_remainder=pipe_reader_pad_remainder,
                                                        num_slices=self.num_instances,
                                                        slice_index=self.instance_id,
                                                        device='cpu'
                                                        )
        else:
            raise ValueError(
                "Unet3dMediaPipe: pipe {} not supported!".format(self.pipeName))

    def definegraph(self):
        """
        Method defines the media graph.

        :returns : output images, labels
        """
        if (self.pipeName == "TrainPipeline"):

            img, lbl = self.inputxy()

            img, lbl, coord = self.rand_bias_crop(img, lbl)  # ToDo: coord

            if self.augment == True:

                if self.enable_zoom == True:
                    crop_size_zoom = self.crop_size()
                    img, lbl = self.zoom(img, lbl, crop_size_zoom)
                img = self.cast_bf16(img)
                # ------------H Flip------------------------------
                img, lbl = self.img_reshape_hflip(img), self.lbl_reshape_hflip(
                    lbl)  # reshape output [W,H,D*C,N]
                is_hflip = self.is_hflip()
                img, lbl = self.img_hflip(
                    img, is_hflip), self.lbl_hflip(lbl, is_hflip)

                # ------------V Flip------------------------------
                is_vflip = self.is_vflip()
                img, lbl = self.img_vflip(
                    img, is_vflip), self.lbl_vflip(lbl, is_vflip)

                if self.dim == 3:
                    # ------------D Flip------------------------------
                    is_dflip = self.is_dflip()
                    img, lbl = self.img_reshape_dflip(
                        img), self.lbl_reshape_dflip(lbl)  # reshape output [W*H,D,C,N]
                    img, lbl = self.img_dflip(
                        img, is_dflip), self.lbl_dflip(lbl, is_dflip)

                    # reshape output [W,H,D*C,N]
                    img = self.img_reshape_noise(img)

                # ------------Gaussian noise------------------------------
                noise = self.rnd_normal(self.stddev(), self.seed())
                img = self.add(img, noise)

                # ------------Gaussian blur------------------------------
                img = self.img_reshape_transpose_pre_blur(img)
                gb_sigma = self.gaussian_kernel_sigma()
                gb_sigma = self.gaussian_kernel_apply_blur(gb_sigma)
                img = self.gaussian_blur(img, gb_sigma)

                # ------------Brightness------------------------------
                img = self.img_reshape_brightness(
                    img)  # reshape output  [W,H,D*C,N]
                brt_in = self.brt_in()
                img = self.brightness(img, brt_in)
                # ------------Contrast------------------------------
                img = self.img_reshape_contrast_input(img)  # [W,H,D,C,N]
                scale = self.scale()
                scale = self.cast_bf16(scale)
                min, min_i = self.min(img)
                max, max_i = self.max(img)
                img = self.img_reshape_before_mul(img)
                img = self.mul(img, scale)
                img = self.img_reshape_after_mul(img)
                img = self.clamp(img, min, max)
            # ------------Reshape------------------------------
            img, lbl = self.img_reshape_output(img), self.lbl_reshape_output(
                lbl)  # reshape output [W,H,D,C,N]

            if self.dim == 2:
                # ------------Transpose------------------------------
                # [W,H,D,C,N] -> [W,H,C,D,N]
                img, lbl = self.img_transpose(img), self.lbl_transpose(lbl)
            return img, lbl

        # elif (self.pipeName == "EvalPipeline"):
        # ToDo: needs to be updated once eval supported by mediapipe

        elif (self.pipeName == "BenchmarkPipeline_Train"):

            img, lbl = self.inputxy()

            img = self.crop_img(img)
            lbl = self.crop_lbl(lbl)
            img = self.cast_bf16(img)
            img, lbl = self.img_reshape_output(
                img), self.lbl_reshape_output(lbl)  # reshape output [W,H,D,C,N]

            if self.dim == 2:
                # [W,H,D,C,N] -> [W,H,C,D,N]
                img, lbl = self.img_transpose(img), self.lbl_transpose(lbl)
            return img, lbl

        elif (self.pipeName == "EvalPipeline"):
            img = self.inputX()
            lbl = self.inputY()
            return img, lbl
        elif (self.pipeName == "TestPipeline"):
            img = self.inputX()
            meta = self.inputMeta()
            return img, meta
        else:
            raise ValueError(
                "Unet3dMediaPipe: pipe {} not supported!".format(self.pipeName))


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
                        priv_params: private params for random_flip_func
                                     prob: probability for flip
        """
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.seed = params['seed'] + params['unique_number']
        self.priv_params = params['priv_params']
        self.prob = self.priv_params['prob']
        self.rng = np.random.default_rng(self.seed)

    def __call__(self):
        """
        Callable class method.

        :returns : randomly generated binary output per image.
        """
        a = self.rng.choice([0, 1],
                            p=[(1-self.prob), self.prob],
                            size=self.np_shape)
        a = np.array(a, dtype=self.np_dtype)
        return a


class brightness_func(media_function):
    """
    Class to randomly generate brightness.

    """

    def __init__(self, params):
        """
        Constructor method.

        :params params: brightness_func specific params.
                        shape: output shape
                        dtype: output data type
                        seed: seed to be used
                        priv_params: private params for brightness_func
                                     factor_min: min brightness factor.
                                     factor_max: max brightness factor.
                                     prob: probability for brightness.
                                     batch_size: batch size.
        """
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.seed = params['seed'] + params['unique_number']
        self.priv_params = params['priv_params']
        self.factor_min = self.priv_params['factor_min']
        self.factor_max = self.priv_params['factor_max']
        self.prob = self.priv_params['prob']
        self.batch_size = self.np_shape[0]
        self.rng = np.random.default_rng(self.seed)

    def __call__(self):
        """
        Callable class method.

        :returns : randomly generated brightness
        """
        a = self.rng.choice([0, 1],
                            p=[(1-self.prob), self.prob],
                            size=[self.batch_size])
        brt_val = self.rng.uniform(low=self.factor_min,
                                   high=self.factor_max,
                                   size=self.np_shape)
        brt_val[a == 0] = 1
        brt_val = np.array(brt_val, dtype=self.np_dtype)
        return brt_val


class random_seed_func(media_function):
    """
    Class to randomly generate seed.

    """

    def __init__(self, params):
        """
        Constructor method.

        :params params: random_seed_func specific params.
                        shape: output shape
                        dtype: output data type
                        seed: seed to be used
        """
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.seed = params['seed'] + params['unique_number']
        self.rng = np.random.default_rng(self.seed)

    def __call__(self):
        """
        Callable class method.

        :returns : randomly generated seed value
        """
        a = self.rng.uniform(size=self.np_shape)
        a = np.array(a, dtype=self.np_dtype)
        return a


class random_stddev_func(media_function):
    """
    Class to randomly generate std dev.

    """

    def __init__(self, params):
        """
        Constructor method.

        :params params: random_stddev_func specific params.
                        shape: output shape
                        dtype: output data type
                        seed: seed to be used
                        priv_params: private params for random_stddev_func
                                     min_std_dev: min std_dev.
                                     max_std_dev: max std_dev.
                                     prob: probability for stddev.
        """
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.seed = params['seed'] + params['unique_number']
        self.priv_params = params['priv_params']
        self.prob = self.priv_params['prob']
        self.min_std_dev = self.priv_params['min_std_dev']
        self.max_std_dev = self.priv_params['max_std_dev']
        # self.batch_size = self.priv_params['batch_size']
        self.rng = np.random.default_rng(self.seed)

    def __call__(self):
        """
        Callable class method.

        :returns : randomly generated stddev value
        """
        a = self.rng.choice([0, 1],
                            p=[(1-self.prob), self.prob],
                            size=self.np_shape)

        std_dev = self.rng.uniform(low=self.min_std_dev,
                                   high=self.max_std_dev,
                                   size=self.np_shape)

        std_dev = std_dev * a
        std_dev = np.array(std_dev, dtype=self.np_dtype)
        return std_dev


def get_pad_params(W, H, kW, kH, S=1):
    """
    Method to generate pad params for convolution to give output shape same as input.

    """
    # compute Pad values
    # (padT+padB) = (H * S - S - H + KH)
    padTB = H * S - S - H + kH
    # padL + padR = (W * S - S - W + KW)
    padLR = W * S - S - W + kW
    padL = padLR//2
    padR = padLR - padL
    padT = padTB//2
    padB = padTB - padT
    return padL, padR, padT, padB


class gaussian_kernel_sigma(media_function):
    """
    Class to randomly generate sigma for gaussian kernel.

    """

    def __init__(self, params):
        self.priv_params = params['priv_params']
        self.batch_size = self.priv_params['batch_size']
        self.min_sigma = self.priv_params['min_sigma']
        self.max_sigma = self.priv_params['max_sigma']
        self.seed = params['seed'] + params['unique_number']
        self.rng = np.random.default_rng(self.seed)

    def __call__(self):
        sigmas = self.rng.uniform(
            low=self.min_sigma, high=self.max_sigma, size=(self.batch_size))
        return [sigmas]


class gaussian_kernel_apply_blur(media_function):
    """
    Class to randomly generate apply probability for gaussian kernel.

    """

    def __init__(self, params):
        self.priv_params = params['priv_params']
        self.batch_size = self.priv_params['batch_size']
        self.seed = params['seed'] + params['unique_number']
        self.rng = np.random.default_rng(self.seed)
        self.prob = self.priv_params['prob']

    def __call__(self, sigmas):
        apply_blur = self.rng.choice(
            [0, 1], p=[(1-self.prob), self.prob], size=self.batch_size)
        apply_blur = apply_blur * sigmas
        return [apply_blur]


class gaussian_kernel_func(media_function):
    """
    Class to randomly generate gaussian kernel for convolution.

    """

    def __init__(self, params):
        """
        Constructor method.

        :params params: gaussian_kernel_func specific params.
                        priv_params: private params for gaussian_kernel_func
                                     batch_size: batch size.
                                     channels: number of channels of image.
                                     min_sigma: min sigma for gaussian random values.
                                     max_sigma: max sigma for gaussian random values.
                                     input_depth: depth of gaussian kernel.
        """
        self.priv_params = params['priv_params']
        self.batch_size = self.priv_params['batch_size']
        self.channels = self.priv_params['channels']
        self.min_sigma = self.priv_params['min_sigma']
        self.max_sigma = self.priv_params['max_sigma']
        self.input_depth = self.priv_params['input_depth']
        self.kSize = int(2 * math.ceil(3 * self.max_sigma) + 1)

    def __call__(self, sigmas, apply_blur):
        """
        Callable class method.

        :returns : randomly generated gaussian kernel
        """
        a = self.create_gaussian_kernel(sigmas, apply_blur)
        return a

    def create_gaussian_kernel(self, sigmas, apply_blur):
        """
        method to generate gaussian kernel.

        :returns : randomly generated gaussian kernel
        """
        gaussianWeights = self.create_oneD_gaussian_kernel(sigmas, apply_blur)
        gaussianWeights_np = np.array(gaussianWeights, dtype=np.float32)
        gaussianWeights_np = np.transpose(gaussianWeights_np)
        gaussianWeights_np = np.tile(
            gaussianWeights_np, self.channels*self.input_depth)
        gaussianWeights_np = np.expand_dims(gaussianWeights_np, axis=0)
        gaussianWeights_np = np.expand_dims(gaussianWeights_np, axis=2)
        return gaussianWeights_np

    def create_oneD_gaussian_kernel(self, sigmas, apply_blur):
        """
        method to compute 1D gaussian kernel of shape based on current sigma

        :returns :  randomly generated 1D gaussian kernel
        """
        maxSizeOneD = self.kSize
        gaussianWeights = []
        for sigma, apply in zip(sigmas, apply_blur):
            if(apply == 0):  # Do not blur
                weightG = [0.0]*maxSizeOneD
                mid = maxSizeOneD//2
                weightG[mid] = 1.0
                gaussianWeights.append(weightG)
            else:
                if sigma < self.min_sigma:
                    sigma = self.min_sigma
                sizeOneD = 2 * math.ceil(3 * sigma) + 1
                r = int((sizeOneD - 1) / 2)
                exp_scale = 0.5 / (sigma * sigma)
                sum = 0.0
                # Calculate first half
                weightG = [0.0]*sizeOneD

                for x in range(-r, 0):
                    weightG[x+r] = math.exp(-(x * x * exp_scale))
                    sum += weightG[x+r]

                # Total sum, it's symmetric with `1` in the center.
                sum *= 2.0
                sum += 1.0
                scale = 1.0 / float(sum)
                # place center, scaled element
                weightG[r] = scale
                # scale all elements so they sum up to 1, duplicate the second half
                for x in range(0, r):
                    weightG[x] *= scale
                    weightG[2 * r - x] = weightG[x]
                # make length equal to max length
                lenDiff = maxSizeOneD - len(weightG)
                if(lenDiff > 0):
                    halfLenDiff = int(lenDiff/2)
                    prefixZeros = [0]*halfLenDiff
                    suffixZeros = [0]*(lenDiff-halfLenDiff)
                    weightG = prefixZeros+weightG+suffixZeros
                gaussianWeights.append(weightG)
        return gaussianWeights


class random_scale_func(media_function):
    """
    Class to randomly generate scale for Contrast.

    """

    def __init__(self, params):
        """
        Constructor method.

        :params params: random_scale_func specific params.
                        shape: output shape
                        dtype: output data type
                        seed: seed to be used
                        priv_params: private params for random_scale_func
                                     scale_min: min scale.
                                     scale_max: max scale.
                                     prob: probability for scale.
        """
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.seed = params['seed'] + params['unique_number']
        self.priv_params = params['priv_params']
        self.scale_min = self.priv_params['scale_min']
        self.scale_max = self.priv_params['scale_max']
        self.prob = self.priv_params['prob']
        self.batch_size = self.np_shape[0]
        self.rng = np.random.default_rng(self.seed)

    def __call__(self):
        """
        Callable class method.

        :returns : randomly generated stddev value
        """
        a = self.rng.choice([0, 1],
                            p=[(1-self.prob), self.prob],
                            size=[self.batch_size])
        s = self.rng.uniform(low=self.scale_min,
                             high=self.scale_max, size=self.np_shape)
        s[a == 0] = 1
        s = np.array(s, dtype=self.np_dtype)
        return s


class random_zoom_func(media_function):
    """
    Class to randomly generate crop shape for zoom.

    """

    def __init__(self, params):
        """
        Constructor method.

        :params params: random_zoom_func specific params.
                        shape: output shape
                        dtype: output data type
                        seed: seed to be used
                        priv_params: private params for random_zoom_func
                                     crop_min: min crop factor.
                                     crop_max: max crop factor.
                                     prob: probability for crop.
                                     patch_size : patch size.
        """
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.batch_size = self.np_shape[0]
        self.seed = params['seed'] + params['unique_number']
        self.priv_params = params['priv_params']
        self.crop_min = self.priv_params['crop_min']
        self.crop_max = self.priv_params['crop_max']
        self.prob = self.priv_params['prob']
        self.patch_size = self.priv_params['patch_size'].copy()
        self.patch_size_ar = np.array(self.patch_size, dtype=self.np_dtype)
        self.rng = np.random.default_rng(self.seed)

    def __call__(self):
        """
        Callable class method.

        :returns : randomly generated crop shape
        """
        apply_zoom = self.rng.choice(
            [0, 1], p=[(1-self.prob), self.prob], size=[self.batch_size])
        crop_factor = self.rng.uniform(
            low=self.crop_min, high=self.crop_max, size=[self.batch_size])
        # crop_factor = np.array(crop_factor, dtype=dt.FLOAT32)

        cropped_patch_ar = np.zeros(self.np_shape, dtype=self.np_dtype)

        for i in range(self.batch_size):
            if apply_zoom[i] == 1:
                cropped_patch_ar[i] = np.array(
                    self.patch_size_ar * crop_factor[i], dtype=self.np_dtype)
            else:
                cropped_patch_ar[i] = self.patch_size_ar

        return cropped_patch_ar
