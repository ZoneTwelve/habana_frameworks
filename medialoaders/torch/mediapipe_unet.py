#!/bin/env python
import numpy as np
import time
import glob
import matplotlib.pyplot as plt
from habana_frameworks.mediapipe.operators.cpu_nodes.cpu_nodes import media_function
from habana_frameworks.mediapipe import fn  # NOQA
from habana_frameworks.mediapipe.mediapipe import MediaPipe  # NOQA
from habana_frameworks.mediapipe.media_types import imgtype as it  # NOQA
from habana_frameworks.mediapipe.media_types import dtype as dt  # NOQA

flip_priv_params = {
    'prob': 1/3,
}

brt_priv_params = {
    'factor': 0.3,
    'prob': 0.1,
}

noise_priv_params = {
    'prob': 0.1,
    'std_dev': 0.1,
}

g_in_buf0 = []
g_in_buf1 = []

probe_node_params = {
    'node': 0,
}


class UnetMlPerfMediaPipe(MediaPipe):
    def __init__(self, device, queue_depth, batch_size, input_list, patch_size, seed, drop_remainder, num_slices, slice_index, num_threads=1, is_testing=False):
        super().__init__(device=device,
                         prefetch_depth=queue_depth,
                         batch_size=batch_size,
                         num_threads=num_threads,
                         pipe_name=self.__class__.__name__)
        if(seed == None):
            seed = int(time.time_ns() % (2**31 - 1))
        print("media data loader seed : ", seed)

        self.is_testing = is_testing
        self.batch_size = batch_size
        self.patch_size = patch_size.copy()

        # reader
        self.inputxy = fn.ReadNumpyDatasetFromDir(num_outputs=2,
                                                  shuffle=True,
                                                  file_list=input_list,
                                                  dtype=[dt.FLOAT32, dt.UINT8],
                                                  dense=False,
                                                  seed=seed,
                                                  num_slices=num_slices,
                                                  slice_index=slice_index,
                                                  drop_remainder=drop_remainder,
                                                  num_readers=4)
        if(self.is_testing):
            pnp = probe_node_params.copy()
            pnp['node'] = 0
            self.capture0 = fn.MediaFunc(func=buf_probe_func,
                                         shape=[0, 0, 0, batch_size],
                                         dtype=dt.FLOAT32,
                                         priv_params=pnp)
            pnp = probe_node_params.copy()
            pnp['node'] = 1
            self.capture1 = fn.MediaFunc(func=buf_probe_func,
                                         shape=[0, 0, 0, batch_size],
                                         dtype=dt.UINT8,
                                         priv_params=pnp)
        # random biased crop
        self.rand_bias_crop = fn.RandomBiasedCrop(patch_size=self.patch_size,
                                                  over_sampling=0.4,
                                                  seed=seed,
                                                  num_workers=4, cache_bboxes=True)

        # squeeze
        shape = self.patch_size.copy()
        shape.append(self.batch_size)
        self.img_sqz = fn.Reshape(size=shape,
                                  tensorDim=len(shape),
                                  layout='',
                                  dtype=dt.FLOAT32)
        self.lbl_sqz = fn.Reshape(size=shape,
                                  tensorDim=len(shape),
                                  layout='',
                                  dtype=dt.UINT8)

        # random horizontal flip node
        self.is_hflip = fn.MediaFunc(func=random_flip_func,
                                     shape=[batch_size],
                                     dtype=dt.UINT8,
                                     seed=seed,
                                     priv_params=flip_priv_params)
        self.img_hflip = fn.RandomFlip(horizontal=1, dtype=dt.FLOAT32)
        self.lbl_hflip = fn.RandomFlip(horizontal=1, dtype=dt.UINT8)

        # random vertical flip node
        self.is_vflip = fn.MediaFunc(func=random_flip_func,
                                     shape=[batch_size],
                                     seed=seed,
                                     dtype=dt.UINT8,
                                     priv_params=flip_priv_params)
        self.img_vflip = fn.RandomFlip(vertical=1, dtype=dt.FLOAT32)
        self.lbl_vflip = fn.RandomFlip(vertical=1, dtype=dt.UINT8)

        # random depth flip node
        # img
        shape = self.patch_size.copy()
        shape.append(self.batch_size)
        shape_3d = [shape[0] * shape[1], shape[2], 1, shape[3]]
        self.is_dflip = fn.MediaFunc(func=random_flip_func,
                                     shape=[batch_size],
                                     seed=seed,
                                     dtype=dt.UINT8,
                                     priv_params=flip_priv_params)
        self.img_dflip_reshape3d_op = fn.Reshape(size=shape_3d,
                                                 tensorDim=len(shape_3d),
                                                 layout='',
                                                 dtype=dt.FLOAT32)
        self.img_dflip = fn.RandomFlip(vertical=1, dtype=dt.FLOAT32)
        self.img_dflip_reshape4d_op = fn.Reshape(size=shape,
                                                 tensorDim=len(shape),
                                                 layout='',
                                                 dtype=dt.FLOAT32)
        # lbl
        self.lbl_dflip_reshape3d_op = fn.Reshape(size=shape_3d,
                                                 tensorDim=len(shape_3d),
                                                 layout='',
                                                 dtype=dt.UINT8)
        self.lbl_dflip = fn.RandomFlip(vertical=1, dtype=dt.UINT8)
        shape = self.patch_size.copy()
        shape.append(self.batch_size)
        self.lbl_dflip_reshape4d_op = fn.Reshape(size=shape,
                                                 tensorDim=len(shape),
                                                 layout='',
                                                 dtype=dt.UINT8)

        # random brightness
        self.brt_in = fn.MediaFunc(func=brightness_func,
                                   seed=seed,
                                   shape=[batch_size],
                                   dtype=dt.FLOAT32,
                                   priv_params=brt_priv_params)
        self.brightness = fn.Brightness(dtype=dt.FLOAT32)

        # Gaussian noise
        shape = self.patch_size.copy()
        shape.append(self.batch_size)
        self.seed = fn.MediaFunc(func=random_seed_func,
                                 shape=[1],
                                 dtype=dt.UINT32,
                                 seed=seed)
        self.stddev = fn.MediaFunc(func=random_stddev_func,
                                   shape=[batch_size],
                                   dtype=dt.FLOAT32,
                                   seed=seed,
                                   priv_params=noise_priv_params)
        self.rnd_normal = fn.RandomNormal(mean=0.0,
                                          stddev=0.1,
                                          dtype=dt.FLOAT32,
                                          dims=len(shape),
                                          shape=shape)
        self.add = fn.Add(dtype=dt.FLOAT32)

        # unsqueeze
        shape = self.patch_size.copy()
        shape.append(1)
        shape.append(self.batch_size)
        self.img_unsqz = fn.Reshape(size=shape,
                                    tensorDim=len(shape),
                                    layout='',
                                    dtype=dt.FLOAT32)
        self.lbl_unsqz = fn.Reshape(size=shape,
                                    tensorDim=len(shape),
                                    layout='',
                                    dtype=dt.UINT8)

    def definegraph(self):
        img, lbl = self.inputxy()
        if(self.is_testing):
            img, lbl = self.capture0(img), self.capture1(lbl)

        # biased crop
        img, lbl, coord = self.rand_bias_crop(img, lbl)

        # squeeze
        img, lbl = self.img_sqz(img), self.lbl_sqz(lbl)

        # random H flip
        is_hflip = self.is_hflip()
        img, lbl = self.img_hflip(img, is_hflip), self.lbl_hflip(lbl, is_hflip)

        # random V flip

        is_vflip = self.is_vflip()
        img, lbl = self.img_vflip(img, is_vflip), self.lbl_vflip(lbl, is_vflip)

        # random D flip
        is_dflip = self.is_dflip()
        img, lbl = self.img_dflip_reshape3d_op(
            img), self.lbl_dflip_reshape3d_op(lbl)
        img, lbl = self.img_dflip(img, is_dflip), self.lbl_dflip(lbl, is_dflip)
        img, lbl = self.img_dflip_reshape4d_op(
            img), self.lbl_dflip_reshape4d_op(lbl)

        # random brightness
        brt_in = self.brt_in()
        img = self.brightness(img, brt_in)

        # random normal
        noise = self.rnd_normal(self.stddev(), self.seed())
        img = self.add(img, noise)

        # unsqueeze
        img, lbl = self.img_unsqz(img), self.lbl_unsqz(lbl)

        if(self.is_testing):
            return img, lbl, coord, is_hflip, is_vflip, is_dflip, brt_in, noise
        else:
            return img, lbl


class buf_probe_func(media_function):
    def __init__(self, params):
        self.priv_params = params['priv_params']
        self.node = self.priv_params['node']

    def __call__(self, input):
        global g_in_buf0
        global g_in_buf1
        if(self.node == 0):
            g_in_buf0.append(input)
        else:
            g_in_buf1.append(input)
        return [input]


class random_flip_func(media_function):
    def __init__(self, params):
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.seed = params['seed'] + params['unique_number']
        self.priv_params = params['priv_params']
        self.prob = self.priv_params['prob']
        self.rng = np.random.default_rng(self.seed)

    def __call__(self):
        a = self.rng.choice([0, 1],
                            p=[(1-self.prob), self.prob],
                            size=self.np_shape)
        a = np.array(a, dtype=self.np_dtype)
        return a


class brightness_func(media_function):
    def __init__(self, params):
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.seed = params['seed'] + params['unique_number']
        self.priv_params = params['priv_params']
        self.factor = self.priv_params['factor']
        self.prob = self.priv_params['prob']
        self.rng = np.random.default_rng(self.seed)

    def __call__(self):
        a = self.rng.choice([0, 1],
                            p=[(1-self.prob), self.prob],
                            size=self.np_shape)
        brt_val = self.rng.uniform(1-self.factor,
                                   1+self.factor,
                                   self.np_shape)
        brt_val = brt_val * a
        brt_val = 1 + brt_val
        brt_val = np.array(brt_val, dtype=self.np_dtype)
        return brt_val


class random_seed_func(media_function):
    def __init__(self, params):
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.seed = params['seed'] + params['unique_number']
        self.rng = np.random.default_rng(self.seed)

    def __call__(self):
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
        """
        self.np_shape = params['shape'][::-1]
        self.np_dtype = params['dtype']
        self.seed = params['seed'] + params['unique_number']
        self.priv_params = params['priv_params']
        self.prob = self.priv_params['prob']
        self.std_dev = self.priv_params['std_dev']
        self.rng = np.random.default_rng(self.seed)

    def __call__(self):
        """
        Callable class method.

        :returns : randomly generated stddev value
        """
        a = self.rng.choice([0, 1],
                            p=[(1-self.prob), self.prob],
                            size=self.np_shape)

        std_dev = self.rng.uniform(low=0.0,
                                   high=self.std_dev,
                                   size=self.np_shape)

        std_dev = std_dev * a
        std_dev = np.array(std_dev, dtype=self.np_dtype)
        return std_dev


def validate_hpu_node(img_in, lbl_in, coord, is_hflip, is_vflip,
                      is_dflip, brt_in, noise, res_img, res_lbl):

    img = img_in[:,
                 coord[0]:coord[1],
                 coord[2]:coord[3],
                 coord[4]:coord[5]]
    lbl = lbl_in[:,
                 coord[0]:coord[1],
                 coord[2]:coord[3],
                 coord[4]:coord[5]]

    if(is_hflip == 1):
        img = np.flip(img, axis=-1)
        lbl = np.flip(lbl, axis=-1)

    if(is_vflip == 1):
        img = np.flip(img, axis=-2)
        lbl = np.flip(lbl, axis=-2)

    if(is_dflip == 1):
        img = np.flip(img, axis=-3)
        lbl = np.flip(lbl, axis=-3)
    if(brt_in == 0):
        raise ValueError("Brightness cannot be zero")
    img = img * brt_in
    img = img + noise

    if(not np.array_equal(img, res_img)):
        for j in range(img.shape[1]):
            if(not np.array_equal(img[0][j], res_img[0][j])):
                print("Img mismatch in plane", j)
                break
        return -1
    if(not np.array_equal(lbl, res_lbl)):
        for j in range(lbl.shape[1]):
            if(not np.array_equal(lbl[0][j], res_lbl[0][j])):
                print("Lbl mismatch in plane", j)
                for k in range(lbl.shape[2]):
                    if(not np.array_equal(lbl[0][j][k], res_lbl[0][j][k])):
                        print("Lbl mismatch in subplane", k)
                        break
        return -1
    return 0


def validate(bcnt, batch_size, data, x_in, y_in):
    img = data[0]
    img = img.as_cpu().as_nparray()
    lbl = data[1]
    lbl = lbl.as_cpu().as_nparray()
    coord = data[2]
    coord = coord.as_cpu().as_nparray()
    is_hflip = data[3]
    is_hflip = is_hflip.as_cpu().as_nparray()
    is_vflip = data[4]
    is_vflip = is_vflip.as_cpu().as_nparray()
    is_dflip = data[5]
    is_dflip = is_dflip.as_cpu().as_nparray()
    brt_in = data[6]
    brt_in = brt_in.as_cpu().as_nparray()
    noise = data[7]
    noise = noise.as_cpu().as_nparray()
    in_img = g_in_buf0.pop(0)
    in_lbl = g_in_buf1.pop(0)
    for i in range(batch_size):
        x_data = np.load(x_in[i])
        y_data = np.load(y_in[i])
        if(not np.array_equal(x_data, in_img[i])):
            print("numpy reader img array mismatch")
            break
        if(not np.array_equal(y_data, in_lbl[i])):
            print("numpy reader lbl array mismatch")
            break
        if(validate_hpu_node(in_img[i],
                             in_lbl[i],
                             coord[i],
                             is_hflip[i],
                             is_vflip[i],
                             is_dflip[i],
                             brt_in[i],
                             noise[i],
                             img[i],
                             lbl[i]) < 0):
            print("\nMismatch in batch {} - {}".format(bcnt, i))
            exit()
        # print(in_img[i].shape)
        # print(img[i].shape)
        # print(coord[i])
        img_slice = img[i][0, 64, :, :]
        oimg_slice = in_img[i][0, coord[i][0]+64, :, :]
        show_slices([oimg_slice, img_slice])
        plt.show()


def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


def main():
    global g_in_buf0
    global g_in_buf1

    seed = int(time.time_ns() % (2**31 - 1))
    epochs = 1
    is_testing = True
    batch_size = 7
    queue_depth = 3
    drop_remainder = True
    patch_size = [128, 128, 128]
    dir = "/software/data/unet3d/kits19/preprocessed_data/"
    pattern0 = "case_*_x.npy"
    pattern1 = "case_*_y.npy"
    x_in = np.array(sorted(glob.glob(dir + "/{}".format(pattern0))))
    y_in = np.array(sorted(glob.glob(dir + "/{}".format(pattern1))))
    np_idx = np.arange(len(x_in))
    input_list = [x_in, y_in]
    pipe = UnetMlPerfMediaPipe(device='hpu',
                               queue_depth=queue_depth,
                               batch_size=batch_size,
                               input_list=input_list,
                               patch_size=patch_size,
                               seed=seed,
                               drop_remainder=drop_remainder,
                               is_testing=is_testing)
    pipe.build()
    if(is_testing == True):
        rng = np.random.default_rng(seed)
        shuffle_idx = np_idx.copy()
        rng.shuffle(shuffle_idx)
    else:
        shuffle_idx = np_idx.copy()
    x_in_shuffle = x_in[shuffle_idx]
    y_in_shuffle = y_in[shuffle_idx]
    for i in range(epochs):
        pipe.iter_init()
        bcnt = 0
        start_time0 = time.perf_counter()
        while(1):
            try:
                print("\rRunning Batch {}".format(bcnt), end=" ")
                data = pipe.run()
                if(is_testing):
                    idx_start = bcnt*batch_size
                    idx_end = idx_start + batch_size
                    x = x_in_shuffle[idx_start:idx_end]
                    y = y_in_shuffle[idx_start:idx_end]
                    validate(bcnt, batch_size, data, x, y)
                bcnt = bcnt + 1

            except StopIteration:
                break
            # images = images.as_cpu().as_nparray()
            # labels = labels.as_cpu().as_nparray()
            # print(images.shape)
            # print(labels.shape)
        del pipe
        print("")
        end_time0 = time.perf_counter()
        t0 = (end_time0 - start_time0)
        print("time taken = ", t0)
        print("batch count ", bcnt)
        ips = 1 / (t0 / bcnt / batch_size)
        print("ips = ", ips)


if __name__ == "__main__":
    main()
