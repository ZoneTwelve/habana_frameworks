from habana_frameworks.mediapipe.media_types import dtype as dt
from habana_frameworks.mediapipe.media_types import layout as lt
from habana_frameworks.mediapipe.media_types import mediaDeviceType as mdt
import media_pipe_types as mpt  # NOQA
import media_pipe_proxy as mppy  # NOQA
import numpy as np


def media_dtype_to_typestr(in_type):
    tstype = None
    if (in_type == mpt.dType.UINT8):
        tstype = 'u1'
    elif (in_type == mpt.dType.UINT16):
        tstype = 'u2'
    elif (in_type == mpt.dType.UINT32):
        tstype = 'u4'
    elif (in_type == mpt.dType.UINT64):
        tstype = 'u8'
    elif (in_type == mpt.dType.INT8):
        tstype = 'i1'
    elif (in_type == mpt.dType.INT16):
        tstype = 'i2'
    elif (in_type == mpt.dType.INT32):
        tstype = 'i4'
    elif (in_type == mpt.dType.INT64):
        tstype = 'i8'
    # needs special handling
    # elif(in_type == 'bfloat16'):
    # elif(in_type == mpt.dType.BFLOAT16):  #TODO: Check if need to enable
        # nptype = np.float16
    elif (in_type == mpt.dType.FLOAT16):
        tstype = 'f2'
    elif (in_type == mpt.dType.FLOAT32):
        tstype = 'f4'
    else:
        raise ValueError("invalid dtype {}".format(in_type))
    return tstype


def array_from_ptr(pointer, typestr, shape, copy=False,
                   read_only_flag=False):
    if (type(shape) != tuple):
        shape = tuple(shape)
    buff = {'data': (pointer, read_only_flag),
            'typestr': typestr,
            'shape': shape}

    class numpy_holder():
        pass

    holder = numpy_holder()
    holder.__array_interface__ = buff
    return np.array(holder, copy=copy)


def str_to_media_dtype(in_type):
    """
    Method to get convert string dtype to backend media dtype.

    :params in_type: media dtype.
    :returns : backend media dtype.
    """
    mdtype = None
    if (in_type == 'uint8'):
        mdtype = mpt.dType.UINT8
    elif (in_type == 'uint16'):
        mdtype = mpt.dType.UINT16
    elif (in_type == 'uint32'):
        mdtype = mpt.dType.UINT32
    elif (in_type == 'uint64'):
        mdtype = mpt.dType.UINT64
    elif (in_type == 'int8'):
        mdtype = mpt.dType.INT8
    elif (in_type == 'int16'):
        mdtype = mpt.dType.INT16
    elif (in_type == 'int32'):
        mdtype = mpt.dType.INT32
    elif (in_type == 'int64'):
        mdtype = mpt.dType.INT64
    elif (in_type == 'bfloat16'):
        mdtype = mpt.dType.BFLOAT16
    elif (in_type == 'float16'):
        mdtype = mpt.dType.FLOAT16
    elif (in_type == 'float32'):
        mdtype = mpt.dType.FLOAT32
    elif (in_type == 'float64'):
        mdtype = mpt.dType.FLOAT64
    elif (in_type == ''):
        mdtype = mpt.dType.NA
    else:
        raise ValueError("invalid dtype {}".format(in_type))
    return mdtype


def media_dtype_to_str(in_type):
    """
    Method to get convert backend media dtype to string dtype.

    :params in_type: backend media dtype.
    :returns : string media dtype.
    """
    mdtype = None
    if (in_type == mpt.dType.UINT8):
        mdtype = 'uint8'
    elif (in_type == mpt.dType.UINT16):
        mdtype = 'uint16'
    elif (in_type == mpt.dType.UINT32):
        mdtype = 'uint32'
    elif (in_type == mpt.dType.UINT64):
        mdtype = 'uint64'
    elif (in_type == mpt.dType.INT8):
        mdtype = 'int8'
    elif (in_type == mpt.dType.INT16):
        mdtype = 'int16'
    elif (in_type == mpt.dType.INT32):
        mdtype = 'int32'
    elif (in_type == mpt.dType.INT64):
        mdtype = 'int64'
    elif (in_type == mpt.dType.BFLOAT16):
        mdtype = 'bfloat16'
    elif (in_type == mpt.dType.FLOAT16):
        mdtype = 'float16'
    elif (in_type == mpt.dType.FLOAT32):
        mdtype = 'float32'
    elif (in_type == mpt.dType.FLOAT64):
        mdtype = 'float64'
    else:
        raise ValueError("invalid dtype {}".format(in_type))
    return mdtype


def get_str_dtype(in_type):
    """
    Method to get string datatype.

    :params in_type: input dtype.
    :returns : string media dtype.
    """
    string = None
    if (type(in_type) is mpt.dType):
        string = media_dtype_to_str(in_type)
    elif (type(in_type) is str):
        string = in_type
    else:
        raise ValueError("invalid dtype {}".format(in_type))
    return string


def np_to_media_dtype(in_type):
    """
    Method to get convert numpy dtype to backend media dtype.

    :params in_type: numpy dtype.
    :returns : backend media dtype.
    """
    mdtype = None
    if (in_type == np.uint8):
        mdtype = mpt.dType.UINT8
    elif (in_type == np.uint16):
        mdtype = mpt.dType.UINT16
    elif (in_type == np.uint32):
        mdtype = mpt.dType.UINT32
    elif (in_type == np.uint64):
        mdtype = mpt.dType.UINT64
    elif (in_type == np.int8):
        mdtype = mpt.dType.INT8
    elif (in_type == np.int16):
        mdtype = mpt.dType.INT16
    elif (in_type == np.int32):
        mdtype = mpt.dType.INT32
    elif (in_type == np.int64):
        mdtype = mpt.dType.INT64
    # this needs special handling
    # elif(in_type == np.bfloat16):
    #    mdtype = mpt.dType.BFLOAT16
    elif (in_type == np.float16):
        mdtype = mpt.dType.FLOAT16
    elif (in_type == np.float32):
        mdtype = mpt.dType.FLOAT32
    elif (in_type == np.float64):
        mdtype = mpt.dType.FLOAT64
    else:
        raise ValueError("invalid dtype {}".format(in_type))
    return mdtype


def get_media_dtype(in_type):
    """
    Method to get backend media datatype.

    :params in_type: input dtype.
    :returns : backend media datatype.
    """
    mdtype = None
    if (type(in_type) is str):
        mdtype = str_to_media_dtype(in_type)
    elif (isinstance(np.dtype, type(in_type))):
        mdtype = np_to_media_dtype(in_type)
    elif (type(in_type) is mpt.dType):
        mdtype = in_type
    else:
        raise ValueError("invalid dtype {}".format(in_type))
    return mdtype


def str_to_np_dtype(in_type):
    """
    Method to get convert string type to numpy dtype.

    :params in_type: string dtype.
    :returns : numpy dtype.
    """
    nptype = None
    if (in_type == 'uint8'):
        nptype = np.uint8
    elif (in_type == 'uint16'):
        nptype = np.uint16
    elif (in_type == 'uint32'):
        nptype = np.uint32
    elif (in_type == 'uint64'):
        nptype = np.uint64
    elif (in_type == 'int8'):
        nptype = np.int8
    elif (in_type == 'int16'):
        nptype = np.int16
    elif (in_type == 'int32'):
        nptype = np.int32
    elif (in_type == 'int64'):
        nptype = np.int64
    # needs special handling
    # elif(in_type == 'bfloat16'):
    #    nptype = mpt.dType.BFLOAT16
    elif (in_type == 'bfloat16'):
        nptype = np.float16  # this is a workaround since bflaot16 has issue
    elif (in_type == 'float16'):
        nptype = np.float16
    elif (in_type == 'float32'):
        nptype = np.float32
    elif (in_type == 'float64'):
        nptype = np.float64
    elif (in_type == ''):
        nptype = np.void
    else:
        raise ValueError("invalid dtype {}".format(in_type))
    return nptype


def media_to_np_dtype(in_type):
    """
    Method to get convert media backend dtype to numpy dtype.

    :params in_type: media backend dtype.
    :returns : numpy dtype.
    """
    nptype = None
    if (in_type == mpt.dType.UINT8):
        nptype = np.uint8
    elif (in_type == mpt.dType.UINT16):
        nptype = np.uint16
    elif (in_type == mpt.dType.UINT32):
        nptype = np.uint32
    elif (in_type == mpt.dType.UINT64):
        nptype = np.uint64
    elif (in_type == mpt.dType.INT8):
        nptype = np.int8
    elif (in_type == mpt.dType.INT16):
        nptype = np.int16
    elif (in_type == mpt.dType.INT32):
        nptype = np.int32
    elif (in_type == mpt.dType.INT64):
        nptype = np.int64
    # needs special handling
    # elif(in_type == 'bfloat16'):
    # elif(in_type == mpt.dType.BFLOAT16):  #TODO: Check if need to enable
        # nptype = np.float16
    elif (in_type == mpt.dType.FLOAT16):
        nptype = np.float16
    elif (in_type == mpt.dType.FLOAT32):
        nptype = np.float32
    elif (in_type == mpt.dType.FLOAT64):
        nptype = np.float64
    else:
        raise ValueError("invalid dtype {}".format(in_type))
    return nptype


def get_numpy_dtype(in_type):
    """
    Method to get backend media datatype.

    :params in_type: input dtype.
    :returns : numpy datatype.
    """
    nptype = None
    if (type(in_type) is str):
        nptype = str_to_np_dtype(in_type)
    elif (type(in_type) is mpt.dType):
        nptype = media_to_np_dtype(in_type)
    elif (isinstance(np.dtype, type(in_type))):
        nptype = in_type
    else:
        raise ValueError("invalid dtype {}".format(in_type))
    return nptype


def str_to_media_fw_type(in_type):
    """
    Method to get convert string framework type to media backend framework type.

    :params in_type: string framework type.
    :returns : media backend framework type.
    """
    fw_type = ""
    if (in_type == "TF_FW"):
        fw_type = mppy.fwType.TF_FW
    elif (in_type == "SYNAPSE_FW"):
        fw_type = mppy.fwType.SYNAPSE_FW
    elif (in_type == "PYTHON_FW"):
        fw_type = mppy.fwType.PYTHON_FW
    elif (in_type == "PYT_FW"):
        fw_type = mppy.fwType.PYT_FW
    else:
        raise ValueError("invalid fw type {}".format(in_type))
    return fw_type


def get_media_fw_type(in_type):
    """
    Method to get backend media framework type.

    :params in_type: input media framework type.
    :returns : backend framework type.
    """
    fw_type = None
    if (type(in_type) is str):
        fw_type = str_to_media_fw_type(in_type)
    elif (type(in_type) is mppy.fwType):
        fw_type = in_type
    else:
        raise ValueError("invalid fw type {}".format(in_type))
    return fw_type


def is_valid_dtype(in_type):
    """
    Method to check if given dtype is valid or not

    :params in_type: input media dtype:
    :returns : bool result.<True/False>
    """
    if (in_type == dt.UINT8 or in_type == dt.UINT16 or in_type == dt.UINT32 or in_type == dt.UINT64 or
       in_type == dt.INT8 or in_type == dt.INT16 or in_type == dt.INT32 or in_type == dt.INT64 or
       in_type == dt.FLOAT16 or in_type == dt.BFLOAT16 or in_type == dt.FLOAT32 or in_type == dt.FLOAT64 or in_type == dt.NDT):
        return True
    else:
        return False


def is_valid_layout(in_lyt):
    """
    Method to check if given layout is valid or not

    :params in_lyt: input layout:
    :returns : bool result.<True/False>
    """
    if (in_lyt == lt.NA or in_lyt == lt.FHWC or in_lyt == lt.NCHW or in_lyt == lt.NHWC):
        return True
    else:
        return False


def getDeviceIdFromDeviceName(device):
    """
    Method to get device type and device id from device name

    :params device: device name.<hpu:0/hpu:/hpu/gaudi2:0/gaudi2:/gaudi2/greco:0/greco:/greco>
    :returns : device type, device id.
    """
    # "hpu:0", "hpu:", "hpu": device id 0

    if (device == mdt.CPU or device == mdt.MIXED):
        return None, 0

    index = device.find(":")
    if index == -1:
        device_id = 0
        try:
            device_type = device.upper()
        except:
            raise ValueError("Supported device type are Greco, Gaudi2")
    else:
        device_name = device[:index]
        try:
            device_type = device_name.upper()
        except:
            raise ValueError("Supported device types are Greco, Gaudi2")

        if index == (len(device) - 1):
            device_id = 0
        else:
            if device[index+1:].isdigit() == True:
                device_id = int(device[index+1:])
                if device_id != 0:
                    raise ValueError(
                        "hpu device:{} device id not valid".format(device))
            else:
                raise ValueError(
                    "hpu device:{} device id not valid".format(device))
    return device_type, device_id
