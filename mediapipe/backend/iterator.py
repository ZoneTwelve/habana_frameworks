import media_pipe_proxy as mppy
from abc import ABC, abstractmethod


class _MediaIterator(ABC):
    """
    Abstract class for mediapipe iterator.

    """

    def __init__(self, _pipeline):
        """
        Constructor method.

        :params _pipeline: mediapipe.
        """
        assert _pipeline is not None, "No pipeline provided"

        self.pipe = _pipeline
        self.len = _pipeline.get_batch_count()
        # print("Num batches ",  self.len)

    @abstractmethod
    def __next__(self):
        """
        Abtract method.

        """
        pass

    @abstractmethod
    def __iter__(self):
        """
        Abstract method.

        """
        pass

    def __len__(self):
        """
        Method to return number of batches.

        """
        return self.len

    def __del__(self):
        del self.pipe


class HPUGenericIterator(_MediaIterator):
    """
    Class to get output tensors from mediapipe.

    """

    def __init__(self, mediapipe):
        """
        Constructor method.

        :params mediapipe: mediapipe
        """
        mediapipe.set_proxy(mppy.fwType.SYNAPSE_FW, 0)
        mediapipe.build()
        super().__init__(_pipeline=mediapipe)

    def __iter__(self):
        """
        Method to initialize mediapipe iterator.

        :returns : iterator for mediapipe
        """
        self.pipe.iter_init()
        return self

    def __next__(self):
        """
        Method to run mediapipe iterator over one batch of dataset and return the output tensors.

        :returns : output tensors.
        """
        output = self.pipe.run()
        return output
