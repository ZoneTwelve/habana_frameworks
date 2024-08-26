from habana_frameworks.mediapipe.backend.nodes import opnode_tensor_info
from habana_frameworks.mediapipe.operators.media_nodes import MediaReaderNode
from habana_frameworks.mediapipe.media_types import dtype as dt
from habana_frameworks.mediapipe.media_types import lastBatchStrategy as lbs
from habana_frameworks.mediapipe.media_types import readerOutType as ro
from habana_frameworks.mediapipe.backend.utils import get_numpy_dtype, get_media_dtype
from habana_frameworks.mediapipe.media_types import mediaDeviceType as mdt
import numpy as np
import os
import pathlib
import glob
import time


# user defined functions
def gen_class_list(dir, class_list=None):
    """
    Method to get list of classes present.

    """
    if class_list:
        return np.array(class_list)

    data_dir = pathlib.Path(dir)
    return np.array(sorted(
        [item.name for item in data_dir.glob('*') if item.is_dir() == True], key=lambda x: '\ '.join(x.split())))


def gen_image_list(dir, format, file_list=None):
    """
    Method to get list of images present.

    """
    if file_list:
        return np.array(file_list)

    else:
        if not isinstance(format, list):
            if isinstance(format, str):
                format = [format]
            else:
                raise ValueError("format {} not supported!".format(format))

        fnames_list = []
        for ext in format:
            fnames_list.extend(glob.glob(dir + "/*/*.{}".format(ext)))

        return np.array(sorted(fnames_list, key=lambda x: '\ '.join(x.split())))


def gen_label_list(file_list, class_names, meta_dtype, file_classes=None):
    """
    Method to generate labels for images.

    """
    lfl = len(file_list)
    # label_list = np.zeros(shape=[lfl], dtype=np.uint64)
    meta_dtype_np = get_numpy_dtype(meta_dtype)
    label_list = np.zeros(shape=[lfl], dtype=meta_dtype_np)
    # since filelist are ordered we will have use of this
    idx = 0
    i = 0
    for i in range(lfl):
        if file_classes:
            cls_name = file_classes[i]
        else:
            cls_name = os.path.basename(os.path.dirname((file_list[i])))
        while (idx < len(class_names)):
            if not (cls_name == class_names[idx]):
                idx = idx + 1
            else:
                break
        if (idx >= len(class_names)):
            raise RuntimeError("optimization error")
        label_list[i] = idx
    return label_list


def get_max_file(img_list, file_sizes=None):
    """
    Getter method to get max file in the image list.

    """
    if file_sizes:
        return max(img_list, key=lambda x: file_sizes[x])
    return max(img_list, key=lambda x: os.stat(x).st_size)


def roundup_filelist_labellist(rng, img_list, lbl_list, round_upto, pad_remainder):
    """
    Method to round up file list and label list.

    """
    num_imgs = len(img_list)
    num_lbl = len(lbl_list)
    if (num_imgs != num_lbl):
        raise ValueError("label and image count not matching")
    append_cnt = int((num_imgs + round_upto - 1) /
                     round_upto) * round_upto - num_imgs
    if (pad_remainder == False):
        idx = rng.choice(
            num_imgs, size=append_cnt, replace=False)
        idx = sorted(idx)
    else:
        idx = np.zeros(shape=(append_cnt), dtype=lbl_list.dtype)
        idx = idx + num_imgs - 1
    img_list_pad = img_list[idx]
    lbl_list_pad = lbl_list[idx]
    img_list = np.append(img_list, img_list_pad)
    lbl_list = np.append(lbl_list, lbl_list_pad)
    return img_list, lbl_list


def rounddown_filelist_labellist(img_list, lbl_list, round_downto):
    """
    Method to round down file list and label list.

    """
    num_imgs = len(img_list)
    num_lbl = len(lbl_list)
    if (num_imgs != num_lbl):
        raise ValueError("label and image count not matching")
    slice_end = int((num_imgs) / round_downto) * round_downto
    if (slice_end == 0):
        raise ValueError("round down failed for img and lbl list")
    img_list = img_list[0: slice_end]
    lbl_list = lbl_list[0: slice_end]
    return img_list, lbl_list


class read_image_from_dir(MediaReaderNode):
    """
    Class defining read image from directory node.

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
        super().__init__(
            name, guid, device, inputs, params, cparams, node_attr, fw_params)
        if (fw_params.device != mdt.LEGACY):
            params["label_dtype"] = get_media_dtype(params["label_dtype"])
            if (params["file_list"] is None):
                params["file_list"] = []
            if (params["class_list"] is None):
                params["class_list"] = []
            if (params["file_sizes"] is None):
                params["file_sizes"] = []
            if (params["file_classes"] is None):
                params["file_classes"] = []
            if (params["max_file"] is None):
                params["max_file"] = ""
            if (params["seed"] is None):
                params["seed"] = -1
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
                    params['shard_once'] = True
            return
        else:
            if params['last_batch_strategy'] != lbs.NONE or params['shard_once'] != None:
                raise ValueError("last_batch_strategy and shard_once not supported in legacy hpu pipe")

        self.batch_size = 1
        self.dir = params['dir']
        self.shuffle = params['shuffle']
        self.seed = params['seed']
        self.max_file = params['max_file']
        self.drop_remainder = params['drop_remainder']
        self.pad_remainder = params['pad_remainder']
        self.format = params['format']
        self.meta_dtype = params["label_dtype"]
        self.num_slices = params['num_slices']
        self.slice_index = params['slice_index']
        self.class_list = params['class_list']
        self.img_list = params['file_list']
        self.file_sizes = params['file_sizes']
        self.file_classes = params['file_classes']

        if (self.seed == None):
            # max supported seed value is 32bit so modulo
            self.seed = int(time.time_ns() % (2**31 - 1))
        self.rng = np.random.default_rng(self.seed)
        print("Finding classes ...", end=" ")
        self.class_list = gen_class_list(self.dir, self.class_list)
        print("Done!")
        self.img_list = gen_image_list(self.dir, self.format, self.img_list)
        # self.img_list = np.resize(self.img_list, (1024*10))
        print("Done!")
        print("Generating labels ...", end=" ")
        self.lbl_list = gen_label_list(
            self.img_list, self.class_list, self.meta_dtype, self.file_classes)
        print("Done!")
        self.num_imgs = len(self.img_list)
        print("Total media files/labels {} classes {}".format(self.num_imgs,
              len(self.class_list)))
        self.iter_loc = 0
        if self.num_imgs == 0:
            raise ValueError("image list is empty")
        self.num_batches = int(self.num_imgs / self.batch_size)
        self.num_imgs_slice = self.num_imgs
        self.img_list_slice = self.img_list
        self.lbl_list_slice = self.lbl_list
        self.num_batches_slice = self.num_batches
        if (self.num_slices < 1):
            raise ValueError("num slice cannot be less then 1")
        if (self.slice_index >= self.num_slices):
            raise ValueError("slice_index cannot be >= num_slices")
        print("num_slices {} slice_index {}".format(
            self.num_slices, self.slice_index))
        print("random seed used ", self.seed)
        self.round_slice_list(self.num_slices)
        # now we slice the dataset
        self.num_imgs_slice = int(self.num_imgs_slice / self.num_slices)
        idx = np.arange(self.num_imgs_slice)
        idx = (idx * self.num_slices) + self.slice_index
        self.img_list_slice = self.img_list_slice[idx]
        self.lbl_list_slice = self.lbl_list_slice[idx]
        print("sliced media files/labels {}".format(self.num_imgs_slice))
        # self.max_file = "/mnt/weka/data/pytorch/imagenet/ILSVRC2012/val/n02130308/ILSVRC2012_val_00033687.JPEG"
        if (self.max_file == None):
            print("Finding largest file ...")
            self.max_file = get_max_file(self.img_list_slice, self.file_sizes)
        print("largest file is ", self.max_file)

        self.batch_size = fw_params.batch_size
        self.round_slice_list(self.batch_size)
        self.num_batches_slice = int(self.num_imgs_slice / self.batch_size)

    def gen_output_info(self):
        """
        Method to generate output type information.

        :returns : output tensor information of type "opnode_tensor_info".
        """
        out_info = []
        o = opnode_tensor_info(dt.NDT,
                               np.array([self.batch_size],
                                        dtype=np.uint32),
                               "")
        out_info.append(o)
        o = opnode_tensor_info(self.meta_dtype,
                               np.array([self.batch_size],
                                        dtype=np.uint32),
                               "")
        out_info.append(o)
        return out_info

    def get_largest_file(self):
        """
        Method to get largest media in the dataset.

        :returns : largest media element in the dataset.
        """
        return self.max_file

    def get_media_output_type(self):
        """
        Method to specify type of media output produced by the reader.

        returns: type of media output which is produced by this reader.
        """
        return ro.FILE_LIST

    def round_slice_list(self, round):
        """
        Method to round up/down.

        :raises ValueError: if mismatch is seen in length of flielist and labellist
        """
        # this function works on sliced dataset only
        if (self.drop_remainder == False):
            self.img_list_slice, self.lbl_list_slice = roundup_filelist_labellist(
                self.rng, self.img_list_slice, self.lbl_list_slice, round, self.pad_remainder)
        else:
            self.img_list_slice, self.lbl_list_slice = rounddown_filelist_labellist(
                self.img_list_slice, self.lbl_list_slice, round)
        self.num_imgs_slice = len(self.img_list_slice)
        if not (len(self.img_list_slice) == len(self.lbl_list_slice)):
            print("{} != {}".format(
                len(self.img_list_slice), len(self.lbl_list_slice)))
            raise ValueError("image list is not same as label list !!!")

    def __len__(self):
        """
        Method to get dataset length.

        returns: length of dataset in units of batch_size.
        """
        return self.num_batches_slice

    def __iter__(self):
        """
        Method to initialize iterator.

        """
        if (self.shuffle == True):
            print("Shuffling ...",  end=" ")
            shuffle_idx = np.arange(self.num_imgs_slice)
            self.rng.shuffle(shuffle_idx)
            self.img_list_slice = self.img_list_slice[shuffle_idx]
            self.lbl_list_slice = self.lbl_list_slice[shuffle_idx]
            print("Done!")
        self.iter_loc = 0
        return self

    def __next__(self):
        """
        Method to get one batch of dataset ouput from iterator.

        """
        if self.iter_loc > (self.num_imgs_slice - 1):
            raise StopIteration
        start = self.iter_loc
        end = self.iter_loc + self.batch_size
        img_list = self.img_list_slice[start:end]
        lbl_list = self.lbl_list_slice[start:end]
        self.iter_loc = self.iter_loc + self.batch_size
        # for i in range(self.batch_size):
        #    print("{} {}".format(i,img_list[i],lbl_list[i]))
        return img_list, lbl_list
