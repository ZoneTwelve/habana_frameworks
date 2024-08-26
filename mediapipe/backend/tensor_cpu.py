import numpy as np
import media_pipe_types as mpt  # NOQA
import media_pipe_nodes as mpn


def media_dtype_to_typestr(in_type):
    tstype = None
    if(in_type == mpt.dType.UINT8):
        tstype = 'u1'
    elif(in_type == mpt.dType.UINT16):
        tstype = 'u2'
    elif(in_type == mpt.dType.UINT32):
        tstype = 'u4'
    elif(in_type == mpt.dType.UINT64):
        tstype = 'u8'
    elif(in_type == mpt.dType.INT8):
        tstype = 'i1'
    elif(in_type == mpt.dType.INT16):
        tstype = 'i2'
    elif(in_type == mpt.dType.INT32):
        tstype = 'i4'
    elif(in_type == mpt.dType.INT64):
        tstype = 'i8'
    # needs special handling
    # elif(in_type == 'bfloat16'):
    # elif(in_type == mpt.dType.BFLOAT16):  #TODO: Check if need to enable
        #nptype = np.float16
    elif(in_type == mpt.dType.FLOAT16):
        tstype = 'f2'
    elif(in_type == mpt.dType.FLOAT32):
        tstype = 'f4'
    elif(in_type == mpt.dType.MEDIA):
        tstype = 'u1'
    else:
        raise ValueError("invalid dtype {}".format(in_type))
    return tstype


def array_from_ptr(base_node, pointer, dType, shape, copy=False,
                   read_only_flag=False):
    typestr = media_dtype_to_typestr(dType)
    if(type(shape) != tuple):
        shape = tuple(shape)
    buff = {'data': (pointer, read_only_flag),
            'typestr': typestr,
            'shape': shape}

    class numpy_holder():
        pass

    holder = numpy_holder()
    holder.__base_node__ = base_node
    holder.__array_interface__ = buff
    return np.array(holder, copy=copy)


class CPUTensor:
    def __init__(self, tensor):
        # print("CPU tensor create : ", flush=True)
        self.name = tensor.name
        self.__tensor = tensor

    def as_nparray(self):
        return np.array(self.__tensor)

    def get_shape(self):
        return self.__tensor.GetShape()

    def get_dtype(self):
        return self.__tensor.GetDtype()

    def __del__(self):
        # data_ptr = self.tensor.GetDataPtr()
        # print("CPU tensor delete :",flush=True)
        self.__tensor.Free()
        del self.__tensor


class HPUTensor:
    def __init__(self, tensor):
        # print("CPU tensor create : ", flush=True)
        self.name = tensor.name
        self.__tensor = tensor

    def as_cpu(self):
        size = self.__tensor.GetSize()
        shape = self.__tensor.GetShape()
        dtype = self.__tensor.GetDtype()
        cputensor = mpn.TensorCPU(self.name + '_py', dtype, 1.0, 0.0, shape)
        cputensor.ToHost(self.__tensor)
        tensor = CPUTensor(cputensor)
        return tensor

    def get_shape(self):
        return self.__tensor.GetShape()

    def get_dtype(self):
        return self.__tensor.GetDtype()

    def get_addr(self):
        return self.__tensor.GetBusAddr()

    def __del__(self):
        # data_ptr = self.tensor.GetDataPtr()
        # print("CPU tensor delete :",flush=True)
        self.__tensor.Free()


def TensorPacker(tensors):
    o = []
    for t in tensors:
        if(t.device == mpn.Device_t.DEVICE_CPU):
            o.append(CPUTensor(t))
        elif(t.device == mpn.Device_t.DEVICE_HPU):
            o.append(HPUTensor(t))
        else:
            raise ValueError("Invalid tensor type received")
    if(len(o) == 1):
        return o[0]
    return o
