import numpy as np


class dataset_shuffler():
    def __init__(self, seed, shuffle, slice_index, num_slices, batch_size,
                 num_unique_elements, drop_remainder, pad_remainder,
                 shuffle_across_dataset, is_mod_slice, pad_across_dataset=False):
        self.rng = np.random.default_rng(seed)
        self.shuffle = shuffle
        self.slice_index = slice_index
        self.num_slices = num_slices
        self.batch_size = batch_size
        self.num_unique_ele = num_unique_elements
        self.drop_remainder = drop_remainder
        self.pad_remainder = pad_remainder
        self.shuffle_across_dataset = shuffle_across_dataset
        self.is_mod_slice = is_mod_slice
        self.pad_across_dataset = pad_across_dataset
        if(self.num_unique_ele < self.num_slices):
            raise ValueError(
                "num of unique data is less the num slice mentioned")
        # first lets handle slices based division
        q = self.num_unique_ele // self.num_slices
        self.num_unique_sliced_idxs = np.zeros(
            self.num_slices, dtype=np.uint64) + q
        if(self.drop_remainder):
            min = int(np.amin(self.num_unique_sliced_idxs))
            self.num_iterable_ele = (min // self.batch_size) * self.batch_size
            self.num_unique_sliced_idxs[:] = self.num_iterable_ele
        else:
            # remainder left
            r = self.num_unique_ele - (q * self.num_slices)
            self.num_unique_sliced_idxs[0:r] += 1
            max = int(np.amax(self.num_unique_sliced_idxs))
            self.num_iterable_ele = (
                (max + self.batch_size - 1) // self.batch_size) * self.batch_size
        if(np.sum(self.num_unique_sliced_idxs) > self.num_unique_ele):
            raise ValueError("Slicing logic exceeding input elements")
        if self.pad_across_dataset:
            # For pad_across_dataset, both shuffle_across_dataset and shuffle should be either False or True
            if (self.shuffle == True) or (self.shuffle_across_dataset == True):
                self.shuffle_across_dataset == True
                self.shuffle = True

            pad_count = int((self.num_iterable_ele * self.num_slices) -
                            np.sum(self.num_unique_sliced_idxs))
            if (pad_count > self.num_unique_ele) and (self.pad_remainder == False):
                raise ValueError(
                    "number of elements to pad > num elements present")
            if self.is_mod_slice == False:
                raise ValueError(
                    "For pad_across_dataset, only modulo sharding is supported")

        if(self.shuffle_across_dataset == True and self.shuffle == False):
            self.shuffle = True
        self.is_cached_sliced_idx = False

    def gen_idx_list(self):
        if self.pad_across_dataset == False:
            if(self.shuffle_across_dataset):
                return self.__gen_shuffle_slice_round_idx_list__()
            else:
                return self.__gen_slice_shuffle_round_idx_list__()
        else:
            return self.__gen_shuffle_round_slice_idx_list__()

    def __gen_shuffle_slice_round_idx_list__(self):
        list_shuffled_idxs = np.arange(self.num_unique_ele)
        if(self.shuffle == True):
            print("Shuffling ...",  end=" ")
            self.rng.shuffle(list_shuffled_idxs)
            print("Done!")
        idx = self.__gen_sliced_idxs__()
        n = int(self.num_unique_sliced_idxs[self.slice_index])
        idx = self.__pad_idx__(idx, n)
        return list_shuffled_idxs[idx]

    def __gen_slice_shuffle_round_idx_list__(self):
        if(self.is_cached_sliced_idx == False):
            self.cached_sliced_idx = self.__gen_sliced_idxs__()
            self.is_cached_sliced_idx = True
        list_shuffled_idxs = np.arange(len(self.cached_sliced_idx),
                                       dtype=np.uint64)
        u = int(self.num_unique_sliced_idxs[self.slice_index])
        if(self.shuffle == True):
            print("Shuffling ...",  end=" ")
            self.rng.shuffle(list_shuffled_idxs[0:u])
            print("Done!")

        list_shuffled_padded_idxs = self.__pad_idx__(list_shuffled_idxs, u)
        return self.cached_sliced_idx[list_shuffled_padded_idxs]

    def __gen_shuffle_round_slice_idx_list__(self):
        list_shuffled_idxs = np.arange(self.num_unique_ele)
        if(self.shuffle == True):
            print("Shuffling ...",  end=" ")
            self.rng.shuffle(list_shuffled_idxs)
            print("Done!")
        idx = np.arange(
            (self.num_iterable_ele * self.num_slices), dtype=np.uint64)
        n = int(np.sum(self.num_unique_sliced_idxs))
        idx = self.__pad_idx__(idx, n)
        idx_slice = self.__gen_sliced_idxs__()
        idx = idx[idx_slice]
        return list_shuffled_idxs[idx]

    def get_num_iterable_elements(self):
        return self.num_iterable_ele

    def __gen_sliced_idxs__(self):
        idx = np.arange(self.num_iterable_ele, dtype=np.uint64)
        if(self.is_mod_slice):
            idx = idx * self.num_slices + self.slice_index
        else:
            if self.pad_across_dataset == False:
                idx = idx + \
                    np.sum(self.num_unique_sliced_idxs[0:self.slice_index])
            else:
                idx += self.num_iterable_ele * self.slice_index
        return idx

    def __pad_idx__(self, idx, n):
        if(self.pad_remainder):
            idx[n:] = idx[n-1]
        else:
            p = len(idx) - n
            idx[n:] = idx[0:p]
        return idx
