# backend code to inject python proxy and get c handle
import media_pipe_proxy as mppx


def set_c_proxy(proxy_device):
    """
    Method to set c proxy in backend.

    :params proxy_device: object holding proxy information.
    """

    new_tensor_dataptr = proxy_device.new_tensor_dataptr
    delete_tensor = proxy_device.delete_tensor

    new_buffer = proxy_device.new_buffer
    delete_buffer = proxy_device.delete_buffer
    get_device_id = proxy_device.get_device_id
    release_device_id = proxy_device.release_device_id
    get_compute_stream = proxy_device.get_compute_stream

    c_proxy = mppx.mediaPythonFwProxy_init(new_tensor_dataptr, delete_tensor,
                                           new_buffer, delete_buffer, get_device_id, release_device_id, get_compute_stream)
    return c_proxy


def del_c_proxy(c_proxy):
    """
    Method to delete c proxy from backend.

    :params c_proxy: c proxy object.
    """
    mppx.mediaPythonFwProxy_del(c_proxy)
