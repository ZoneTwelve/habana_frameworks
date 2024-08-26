# this file will contain abstract python proxy class
# framework will create child class of this andperform actual implementation

from abc import ABC, abstractmethod


class HPUProxy(ABC):
    """
    Abtract class representing HPU Proxy.
    Framworks will derive this class and implement the required functionality.

    """
    @abstractmethod
    def new_buffer(self, size):
        """
        Abtract method for creating new buffer.

        :params size: Buffer size in bytes.
        :returns : device buffer.
        """
        pass

    @abstractmethod
    def delete_buffer(self, data_ptr):
        """
        Abtract method for freeing buffer.

        :params data_ptr: Buffer adrress to be freed.
        """
        pass

    @abstractmethod
    def get_device_id(self, device_type):
        """
        Abtract getter method to get device ID from device type.

        :params device_type: device type.
        :returns : device id.
        """
        pass

    @abstractmethod
    def release_device_id(self, device_id):
        """
        Abtract method to release device.

        :params device_id: device id to release.
        """
        pass

    @abstractmethod
    def get_compute_stream(self):
        """
        Abtract getter method to get device compute stream.

        :returns : device compute stream.
        """
        pass

    @abstractmethod
    def new_tensor_dataptr(self, shape, dtype):
        """
        Abtract method to get device tensor.

        :params shape: shape of tensor.
        :params dtype: datatype of tensor.
        :returns : device tensor data pointer.
        """
        pass

    @abstractmethod
    def delete_tensor(self, data_ptr):
        """
        Abtract method to free allocated tensor.

        :params data_ptr: Buffer adrress to be freed.
        """
        pass

    @abstractmethod
    def flush_tensors(self, tensor_l):
        """
        Abtract method to flush pending tensors in queue.

        :params tensor_l: list of tensors to be flushed.
        """
        pass
