from habana_frameworks.mediapipe.media_types import readerOutType as ro
from ctypes import *
import numpy as np

MEDIA = CDLL("libmedia.so")

media_error = {
    0: "mediaSuccess",
    -1: "mediaFail",
    -2: "mediaInvalidArgument",
    -3: "mediaNotSupported",
    -4: "mediaEndOfFile"
}


class FileList(Structure):
    _fields_ = [('filePath', POINTER(c_char_p))]


class BufferList(Structure):
    _fields_ = [('bufferAddr', POINTER(c_char_p)),
                ('bufferSizes', POINTER(c_uint32))]


class InputTypeList(Union):
    _fields_ = [('fileList', FileList),
                ('bufferList', BufferList)]


class FrameList(Structure):
    _fields_ = [('start', c_uint32),
                ('length', c_uint32)]


class InputList(Structure):
    _fields_ = [('list', InputTypeList),
                ('listSize', c_int)]


class MediaCrop (Structure):
    _fields_ = [('crop_x', c_double),
                ('crop_y', c_double),
                ('crop_width', c_double),
                ('crop_height', c_double),
                ('crop_x_u', c_uint),
                ('crop_y_u', c_uint),
                ('crop_width_u', c_uint),
                ('crop_height_u', c_uint),
                ('isInt', c_bool),
                ('isCropAfterResize', c_bool)]


unknown_error = -len(media_error)
media_success = 0


def is_ok(func, status):
    """
    Method to check backend return status.

    :params func_name: function name as string.
    :params status: return status of backend code.
    :raises RuntimeError: if status is failure.
    """

    if (status <= unknown_error) or (status > 0):
        print("Error Code = ", status)
        raise RuntimeError("Media API {} failed! Error Code = {}.".format(
            func.__name__, "mediaErrorUnknown"))
    elif status != media_success:
        raise RuntimeError("Media API {} failed! Error Code = {}.".format(
            func.__name__, media_error[status]))
    else:
        #logger.LOG_TRACE("{}(): Success.".format(func.__name__))
        pass


class legacy:
    """
    Class defining legacy mediapipe backend interface.

    """

    def __init__(self, batch_size, queue_depth):
        """
        Constructor method.

        :params batch_size: batch size.
        """
        self.batch_size = batch_size
        self.run_hpu_q = []
        self.max_queue_size = queue_depth

    def run_hpu(self, pipeManager, input_list, input_type, rand_crp, is_crp_after_resize, vid_offset, resample_idx, host_buffer, bypass_host_buffers, out_dev_buffer_list, is_gather):
        """
        Method invoke on device execution.

        """
        mediaPipelineRunHpu = MEDIA.mediaPipelineRunHpu
        mediaPipelineRunHpu.restypes = c_int
        mediaPipelineRunHpu.argtypes = [c_void_p,
                                        POINTER(InputList),
                                        POINTER(FrameList),
                                        POINTER(c_void_p),  # resample_idx
                                        POINTER(MediaCrop),
                                        POINTER(c_void_p),
                                        POINTER(c_void_p),
                                        POINTER(c_uint64)]
        il = InputList()
        ctx = None
        crop_param_array = 0
        decoder_node_offset = 0

        if not (input_list is None):
            decoder_node_offset = 1
            itl = InputTypeList()
            if(input_type == ro.FILE_LIST):
                fl_arr = (c_char_p * (len(input_list) + 1))()
                for i in range(len(input_list)):
                    fl_arr[i] = bytes(input_list[i], 'utf-8')
                fl = FileList(fl_arr)
                itl.fileList = fl
            elif(input_type == ro.BUFFER_LIST):
                bl_arr = (c_char_p * (len(input_list) + 1))()
                size_arr = (c_uint32 * (len(input_list) + 1))()
                for i in range(len(input_list)):
                    bl_arr[i] = input_list[i].__array_interface__['data'][0]
                    # INFO: sizes must be multiple of 64
                    # size_arr[i] = ((len(input_list[i]) + 63) & (-64))
                    size_arr[i] = len(input_list[i])
                    #print("{} addr {} len {}".format(i,input_list[i].__array_interface__['data'][0],len(input_list[i])))

                bl = BufferList(bl_arr,
                                size_arr)
                itl.bufferList = bl
                ctx = input_list
            elif(input_type == ro.ADDRESS_LIST):
                bl_arr = (c_char_p * (len(input_list) + 1))()
                size_arr = (c_uint32 * (len(input_list) + 1))()
                for i in range(len(input_list)):
                    addr = bytes(input_list[i][0], 'utf-8')
                    bl_arr[i] = addr
                    size_arr[i] = input_list[i][1]
                    #print("{} addr {} len {}".format(i,bl_arr[i],size_arr[i]))
                bl = BufferList(bl_arr,
                                size_arr)
                itl.bufferList = bl
                ctx = input_list
            else:
                raise ValueError("unsupported input type")

            il = InputList(itl, len(input_list))

        frame_list_array = 0
        if not (vid_offset is None):

            frame_list_array = (FrameList * self.batch_size)()
            for i in range(self.batch_size):
                frame_list_array[i].start = vid_offset[i][0]
                frame_list_array[i].length = vid_offset[i][1]

        resample_idx_array = None
        if not (resample_idx is None):
            resample_idx_array = (c_void_p * self.batch_size)()
            for i in range(self.batch_size):
                pointer = resample_idx[i].__array_interface__['data'][0]
                resample_idx_array[i] = c_void_p(pointer)

        # start_time1 =  time.perf_counter()
        if not (rand_crp is None):
            crop_param_array = (MediaCrop * self.batch_size)()

            if (is_crp_after_resize == True) and (rand_crp.dtype == np.float32):
                raise RuntimeError(
                    "expected uint32 crop window for crop after resize")

            for i in range(self.batch_size):
                crop_param_array[i].isCropAfterResize = is_crp_after_resize
                if (rand_crp.dtype == np.float32):
                    crop_param_array[i].crop_x = c_double(rand_crp[i][0])
                    crop_param_array[i].crop_y = c_double(rand_crp[i][1])
                    crop_param_array[i].crop_width = c_double(rand_crp[i][2])
                    crop_param_array[i].crop_height = c_double(rand_crp[i][3])
                    crop_param_array[i].isInt = False
                else:
                    crop_param_array[i].crop_x_u = rand_crp[i][0]
                    crop_param_array[i].crop_y_u = rand_crp[i][1]
                    crop_param_array[i].crop_width_u = rand_crp[i][2]
                    crop_param_array[i].crop_height_u = rand_crp[i][3]
                    crop_param_array[i].isInt = True

        #start_time2 =  time.perf_counter()
        # INFO: +1 is for accomodating decoder input
        num_host_buffer = len(host_buffer) + decoder_node_offset
        hb_pp = (c_void_p * (num_host_buffer))()
        hb_pp[0] = c_void_p(0)
        gather_node_offset = 0
        if is_gather:
            assert decoder_node_offset == 1, "Error: Gather without decode"
            gather_node_offset = 1
            hb_pp[1] = c_void_p(0)

        for i in range(num_host_buffer - decoder_node_offset - gather_node_offset):
            pointer = host_buffer[i +
                                  gather_node_offset].__array_interface__['data'][0]
            hb_pp[i + decoder_node_offset +
                  gather_node_offset] = c_void_p(pointer)

        num_bypass_buffers = len(bypass_host_buffers)
        byp_hb_pp = (c_void_p * num_bypass_buffers)()
        for i in range(num_bypass_buffers):
            pointer = bypass_host_buffers[i].__array_interface__['data'][0]
            byp_hb_pp[i] = c_void_p(pointer)
        #start_time3 =  time.perf_counter()

        c_dev_buf_list = (c_uint64 * len(out_dev_buffer_list))()
        for i in range(len(out_dev_buffer_list)):
            c_dev_buf_list[i] = c_uint64(out_dev_buffer_list[i])

        self.run_hpu_q.append(il)
        self.run_hpu_q.append(frame_list_array)
        self.run_hpu_q.append(resample_idx)
        self.run_hpu_q.append(resample_idx_array)
        self.run_hpu_q.append(ctx)
        self.run_hpu_q.append(crop_param_array)

        err = mediaPipelineRunHpu(c_void_p(pipeManager),
                                  byref(il),
                                  cast(frame_list_array, POINTER(FrameList)),
                                  resample_idx_array,
                                  cast(crop_param_array, POINTER(MediaCrop)),
                                  hb_pp,
                                  byp_hb_pp,
                                  c_dev_buf_list)
        is_ok(mediaPipelineRunHpu, err)
        #end_time = time.perf_counter()

        # print("================================")
        # print("inside run_hpu")
        # print("Time Elapsed total = " , (end_time - start_time0))
        #print("Time Elapsed filelist = " , (start_time1 - start_time0))
        #print("Time Elapsed crop = " , (start_time2 - start_time1))
        #print("Time Elapsed rtbuffers = " , (start_time3 - start_time2))
        #print("Time Elapsed c api = " , (end_time - start_time3))
        # print("================================")

    def get_output(self):
        """
        Method to getoutput. Here it is responsible for clearing the cache held.

        """
        self.run_hpu_q.pop(0)
        self.run_hpu_q.pop(0)
        self.run_hpu_q.pop(0)
        self.run_hpu_q.pop(0)
        self.run_hpu_q.pop(0)
        self.run_hpu_q.pop(0)

        if(len(self.run_hpu_q) > (6 * self.max_queue_size)):
            raise RuntimeError("Hpu queue overflow")
