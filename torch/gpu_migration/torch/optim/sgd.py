
from . import TorchOptimizersModuleRegister
from torch.optim import SGD
from torch.optim.optimizer import required
from typing import Optional
from habana_frameworks.torch.gpu_migration.torch import TORCH_VERSION

class SGD(SGD, TorchOptimizersModuleRegister):
    """
    .. py:gpumgrcall:: SGD.hpu_modified

    Sets foreach parameter to false.

    """

    is_replace_class = True
    @classmethod
    def _get_class_gpu_migration(cls):
        return getattr(cls.get_module(), cls.__name__)

    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                weight_decay=0, nesterov=False, *, maximize: bool = False, foreach: Optional[bool] = None,
                differentiable: bool = False):
        foreach = False
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach,
                        differentiable=differentiable)
        super().__init__(params, **defaults)
