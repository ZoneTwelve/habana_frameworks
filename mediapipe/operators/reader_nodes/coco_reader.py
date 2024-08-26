from habana_frameworks.mediapipe.operators.media_nodes import MediaReaderNode
from habana_frameworks.mediapipe.media_types import dtype as dt
from habana_frameworks.mediapipe.media_types import lastBatchStrategy as lbs
from habana_frameworks.mediapipe.media_types import readerOutType as ro
from habana_frameworks.mediapipe.backend.nodes import opnode_tensor_info
from habana_frameworks.mediapipe.media_types import mediaDeviceType as mdt
import os
import json
import time
import numpy
import time
import argparse
import numpy as np


def roundup_keylist(rng, img_dict, key_list, round_upto, pad_remainder=False, pad_separate=False):
    """
    Method to round up key list. img dictionary remains same after round up

    """
    # num_imgs = len(img_dict)
    num_key = len(key_list)

    append_cnt = int((num_key + round_upto - 1) /
                     round_upto) * round_upto - num_key

    if(pad_remainder == False):
        idx = rng.choice(num_key, size=append_cnt, replace=False)
        idx = sorted(idx)
    else:
        idx = np.zeros(shape=(append_cnt), dtype=np.uint32)
        idx = idx + num_key - 1

    key_list_pad = key_list[idx]

    if pad_separate == False:
        key_list_pad_ret = np.zeros(shape=(0), dtype=int)
        key_list = np.append(key_list, key_list_pad)
    else:
        key_list_pad_ret = key_list_pad

    return key_list, key_list_pad_ret


def rounddown_keylist(img_dict, key_list, round_downto):
    """
    Method to round down key list and img dictionary.

    """
    num_imgs = len(img_dict)
    num_key = len(key_list)
    if(num_imgs != num_key):
        print("{} != {}".format(num_imgs, num_key))
        raise ValueError("image and key count not matching")

    slice_end = int((num_key) / round_downto) * round_downto
    if(slice_end == 0):
        raise ValueError("round down failed for img and key list")

    key_list = key_list[0: slice_end]
    num_images = len(key_list)
    img_dict_new = {}
    for index in range(num_images):
        image_id = key_list[index]
        img_dict_new[image_id] = img_dict[image_id]

    if not (len(img_dict_new) == len(key_list)):
        print("{} != {}".format(
            len(img_dict_new), len(key_list)))
        raise ValueError(
            "image and key count not matching after rounddown!!!")

    return img_dict_new, key_list


# Implement a datareader for COCO dataset
class coco_reader(MediaReaderNode):
    """
    Class defining coco reader node.

    """

    def __init__(self, name, guid, device, inputs, params, cparams, node_attr, fw_params):
        """
        Constructor method.

        :params name: node name.
        :params guid: guid of node.
        :params device: device on which this node should execute.
        :params params: node specific params.
        :params cparams: backend params.
        :params node_attr: node output information
        """
        super().__init__(name, guid, device, inputs, params, cparams, node_attr,fw_params)

        if (fw_params.device != mdt.LEGACY):
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

        self.annfile = params['annfile']
        self.root = params['root']
        self.slice_index = int(params['slice_index'])
        self.num_slices = int(params['num_slices'])
        self.drop_remainder = params['drop_remainder']
        self.pad_remainder = params['pad_remainder']
        self.seed = params['seed']
        self.shuffle = params['shuffle']
        self.max_file = params['max_file']
        self.partial_batch = params['partial_batch']
        self.max_boxes = 200
        self.ids_dtype = dt.UINT32
        self.sizes_dtype = dt.UINT32
        self.boxes_dtype = dt.FLOAT32
        self.labels_dtype = dt.UINT32
        self.lengths_dtype = dt.UINT32
        self.batch_dtype = dt.UINT32
        self.np_images_shape = np.array([self.batch_size], dtype=np.uint32)
        self.np_ids_shape = np.array([self.batch_size], dtype=np.uint32)
        self.np_sizes_shape = np.array([self.batch_size, 2], dtype=np.uint32)
        self.np_boxes_shape = np.array(
            [self.batch_size, self.max_boxes, 4], dtype=np.uint32)  # this is variable output
        self.np_labels_shape = np.array(
            [self.batch_size, self.max_boxes], dtype=np.uint32)  # this is variable output
        self.np_lengths_shape = np.array(
            [self.batch_size], dtype=np.uint32)
        self.np_batch_shape = np.array([1], dtype=np.uint32)
        if (self.seed == None):
            # max supported seed value is 32bit so modulo
            self.seed = int(time.time_ns() % (2**31 - 1))

        with open(self.annfile) as fp:
            self.data = json.load(fp)

        self.images = {}
        self.label_map = {}

        # 0 stand for the background
        cnt = 0
        for cat in self.data["categories"]:
            cnt += 1
            self.label_map[cat["id"]] = cnt

        # read image attribute
        for img in self.data["images"]:
            img_id = img["id"]
            img_name = img["file_name"]
            img_size = (img["height"], img["width"])
            # print(img_name)
            if img_id in self.images:
                raise Exception("dulpicated image record")
            self.images[img_id] = (img_name, img_size, [])

        # read bboxes
        for bboxes in self.data["annotations"]:
            img_id = bboxes["image_id"]
            category_id = bboxes["category_id"]
            bbox = bboxes["bbox"]
            bbox_label = self.label_map[bboxes["category_id"]]
            self.images[img_id][2].append((bbox, bbox_label))

        # remove images without bbox
        for k, v in list(self.images.items()):
            if len(v[2]) == 0:
                # print("empty image: {}".format(k))
                self.images.pop(k)

        self.keys = numpy.fromiter(self.images.keys(), dtype=int)
        if not (len(self.images) == len(self.keys)):
            print("{} != {}".format(
                len(self.images), len(self.keys)))
            raise ValueError(
                "image and key count not matching!!!")

        if(self.num_slices < 1):
            raise ValueError("num slice cannot be less then 1")
        if(self.slice_index >= self.num_slices):
            raise ValueError("slice_index cannot be >= num_slices")

        self.num_imgs = len(self.images)
        if self.num_imgs == 0:
            raise ValueError("image list is empty")

        # print("coco_reader seed {} shuffle {} drop_remainder {} pad_remainder {}".format(
        #      self.seed, self.shuffle, self.drop_remainder, self.pad_remainder))

        print("coco_reader seed {} shuffle {}".format(self.seed, self.shuffle))
        print("Total images ", self.num_imgs)
        print("num_slices {} slice_index {}".format(
            self.num_slices, self.slice_index))
        self.images_slice = self.images
        self.keys_slice = self.keys
        self.keys_slice_pad = np.zeros(shape=(0), dtype=int)
        self.rng = np.random.default_rng(self.seed)

        self.round_slice_list(self.num_slices, False)

        # now we slice the dataset
        self.num_keys_slice = int(self.num_keys_slice / self.num_slices)
        idx = np.arange(self.num_keys_slice)
        idx = (idx * self.num_slices) + self.slice_index

        self.keys_slice = self.keys_slice[idx]

        img_list_new = {}
        for index in range(self.num_keys_slice):
            image_id = self.keys_slice[index]
            img_list_new[image_id] = self.images_slice[image_id]

        self.images_slice = img_list_new

        print("sliced length {}".format(self.num_keys_slice))

        img_list = []
        if(self.max_file == None):
            print("Finding largest file ...")
            for index in range(self.num_keys_slice):
                image_id = self.keys_slice[index]
                image = self.images_slice[image_id]
                file_name = image[0]
                image_path = os.path.join(self.root, file_name)
                img_list.append(image_path)
            file_max = max(img_list, key=lambda x: os.stat(x).st_size)
            self.max_file = file_max
        print("largest file is ", self.max_file)

        self.batch_size = fw_params.batch_size
        self.np_images_shape = np.array([self.batch_size], dtype=np.uint32)
        self.np_ids_shape = np.array([self.batch_size], dtype=np.uint32)
        self.np_sizes_shape = np.array([self.batch_size, 2], dtype=np.uint32)
        self.np_boxes_shape = np.array(
            [self.batch_size, self.max_boxes, 4], dtype=np.uint32)  # this is variable output
        self.np_labels_shape = np.array(
            [self.batch_size, self.max_boxes], dtype=np.uint32)  # this is variable output
        self.np_lengths_shape = np.array(
            [self.batch_size], dtype=np.uint32)

        if (self.partial_batch == True):
            self.round_slice_list(self.batch_size, True)
        else:
            self.round_slice_list(self.batch_size, False)

        if (self.partial_batch == False) and (len(self.keys_slice_pad) != 0):
            raise ValueError("expected empty pad key list")

        self.num_batches_slice = int(
            (self.num_keys_slice + len(self.keys_slice_pad)) / self.batch_size)

        # print("coco_reader images {} keys {} batches {}  batchsize {}".format(
        #      len(self.images_slice), self.num_keys_slice, self.num_batches_slice, self.batch_size))
        print("coco_reader batches {} batchsize {}".format(
            self.num_batches_slice, self.batch_size))




    def get_largest_file(self):
        """
        Method to get largest media in the dataset.

        returns: largest media element in the dataset.
        """
        return self.max_file

    def get_media_output_type(self):
        """
        Method to specify type of media output produced by the reader.

        returns: type of media output which is produced by this reader.
        """
        return ro.FILE_LIST

    def __len__(self):
        """
        Method to get dataset length.

        returns: length of dataset in units of batch_size.
        """
        return self.num_batches_slice  # self.len

    def __iter__(self):
        """
        Method to initialize iterator.

        """
        if(self.shuffle == True):
            print("Shuffling ...",  end=" ")
            shuffle_idx = np.arange(self.num_keys_slice)
            self.rng.shuffle(shuffle_idx)
            self.keys_slice = self.keys_slice[shuffle_idx]
            print("Done!")
        self.current_index = 0
        return self

    def __next__(self):
        """
        Method to get one batch of dataset ouput from iterator.

        """
        last_index = self.current_index + self.batch_size
        # if last_index > (self.batch_size * self.num_batches_slice):
        if last_index > (self.num_keys_slice + len(self.keys_slice_pad)):
            raise StopIteration

        images = []
        ids = np.zeros(shape=self.np_ids_shape, dtype=self.ids_dtype)
        sizes = np.zeros(shape=self.np_sizes_shape, dtype=self.sizes_dtype)
        lengths = np.zeros(shape=self.np_lengths_shape,
                           dtype=self.lengths_dtype)
        boxes = np.zeros(shape=self.np_boxes_shape, dtype=self.boxes_dtype)
        labels = np.zeros(shape=self.np_labels_shape, dtype=self.labels_dtype)
        batch = np.zeros(shape=self.np_batch_shape,
                         dtype=self.batch_dtype)

        batch[0] = self.batch_size
        if last_index > self.num_keys_slice:
            # partial batch
            batch[0] = self.batch_size - len(self.keys_slice_pad)
        i = 0
        for index in range(self.current_index, last_index):
            if index < self.num_keys_slice:
                image_id = self.keys_slice[index]
            else:
                image_id = self.keys_slice_pad[index - self.num_keys_slice]
            image = self.images_slice[image_id]
            file_name = image[0]
            htot, wtot = image[1]
            image_path = os.path.join(self.root, file_name)

            images.append(image_path)
            ids[i] = image_id
            sizes[i] = [htot, wtot]
            lengths[i] = len(image[2])
            if (lengths[i] > self.max_boxes):
                raise RuntimeError(
                    "Number of boxes in the image are more than 200!")
            j = 0
            for (l, t, w, h), bbox_label in image[2]:
                r = l + w
                b = t + h
                box = [l / wtot, t / htot, r / wtot, b / htot]
                boxes[i, j] = box
                labels[i, j] = bbox_label
                j = j + 1
            i = i + 1

        self.current_index = last_index

        images = np.array(images)
        return images, ids, sizes, boxes, labels, lengths, batch

    def gen_output_info(self):
        """
        Method to generate output type information.

        :returns : output tensor information of type "opnode_tensor_info".
        """
        out_info = []
        o = opnode_tensor_info(dt.NDT, self.np_images_shape[::-1], "")
        out_info.append(o)
        o = opnode_tensor_info(self.ids_dtype, self.np_ids_shape[::-1], "")
        out_info.append(o)
        o = opnode_tensor_info(self.sizes_dtype, self.np_sizes_shape[::-1], "")
        out_info.append(o)
        o = opnode_tensor_info(self.boxes_dtype, self.np_boxes_shape[::-1], "")
        out_info.append(o)
        o = opnode_tensor_info(
            self.labels_dtype, self.np_labels_shape[::-1], "")
        out_info.append(o)
        o = opnode_tensor_info(
            self.lengths_dtype, self.np_lengths_shape[::-1], "")
        out_info.append(o)
        o = opnode_tensor_info(
            self.batch_dtype, self.np_batch_shape[::-1], "")
        out_info.append(o)
        return out_info


    def round_slice_list(self, round, pad_separate):
        """
        Method to round up/down.

        """
        # this function works on sliced dataset only
        if(self.drop_remainder == False):
            self.keys_slice, key_slice_pad = roundup_keylist(
                self.rng, self.images_slice, self.keys_slice, round, self.pad_remainder, pad_separate)
            if (pad_separate == True):
                self.keys_slice_pad = np.append(
                    self.keys_slice_pad, key_slice_pad)
        else:
            self.images_slice, self.keys_slice = rounddown_keylist(
                self.images_slice, self.keys_slice, round)

        self.num_keys_slice = len(self.keys_slice)

        # print("round_slice_list image count {} key list count {} pad key count {} ".format(len(self.images_slice), len(self.keys_slice), len(self.keys_slice_pad)))
