###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import inspect
from collections import defaultdict
from types import ModuleType
from typing import Callable, List


class DictSetOne(dict):
    """Skip assigning value if the key is already present in dict."""

    _visited_obj: set = set()

    def __setitem__(self, __key, __value) -> None:
        if __key not in self._visited_obj:
            self._visited_obj.add(__key)
            return super().__setitem__(__key, __value)
        return None


class BaseModuleRegister(object):
    """Base class of module register.

    Attributes:
        GLOBAL_FUNC_MAP: a DictSetOne of defaultdict. Maintains the original module and gpu_migration module mapping.
        parentFunc: a DictSetOne. Contains the map of the name and original functions.
        is_replace_class: bool. If True, overwrite the original class. Default is False.
    """

    GLOBAL_FUNC_MAP: defaultdict = defaultdict(DictSetOne)
    parentFunc = DictSetOne()
    is_replace_class = False

    def init_func(self) -> List:
        return []

    @classmethod
    def enabled(self) -> bool:
        return True

    @classmethod
    def register_f(cls, func_name=None) -> Callable:
        """Decorator that saving functions in gpu_migration module and their original version.

        Args:
            func_name : str. If func_name is given, saving original function. Default is None.
        """
        if func_name is not None:
            cls.parentFunc[str(id(cls)) + func_name] = getattr(
                cls.get_module(), func_name
            )

        def register_func(func):
            cls.GLOBAL_FUNC_MAP[id(cls)][
                getattr(cls.get_module(), func.__name__)
            ] = func
            return func

        return register_func

    @classmethod
    def register_op(cls) -> Callable:
        """OP register decorator.

        This is a registration only for VariableFunctions. In future there is a need to resign from call_function() in hijacks and unify the method.
        """
        def register_func(func):
            setattr(cls, func.__name__, getattr(cls.get_module(), func.__name__))
            cls.parentFunc[str(id(cls)) + func.__name__] = getattr(cls.get_module(), func.__name__)
            cls.GLOBAL_FUNC_MAP[id(cls)][getattr(cls.get_module(), func.__name__)] = func
            return func
        return register_func

    @classmethod
    def wrap_api(cls, func: Callable) -> Callable:
        """Get the apis which should replace original apis."""
        new_func = cls.GLOBAL_FUNC_MAP[id(cls)][func]
        func_name = func.__name__

        orig_sig = None
        try:
            if hasattr(cls, "_get_class_gpu_migration"):
                orig_sig = inspect.signature(
                    getattr(cls._get_class_gpu_migration(), func_name)
                )
            else:
                orig_sig = inspect.signature(getattr(cls.get_module(), func_name))
        except Exception:
            pass

        if orig_sig:
            new_sig = inspect.signature(new_func)

            if hasattr(cls, "_get_class_gpu_migration"):
                old_func_name = ".".join(
                    [
                        cls.get_module().__name__,
                        cls._get_class_gpu_migration().__name__,
                        func_name,
                    ]
                )
            else:
                old_func_name = ".".join([cls.get_module().__name__, func_name])
            assert str(orig_sig).replace("'","") == str(new_sig).replace(
                "'",""
            ), "The signature in {} is {}, but get {}".format(
                old_func_name, orig_sig, new_sig
            )
        return new_func

    @classmethod
    def get_functions(cls) -> List:
        for func_name in cls._save_orig_func_gpu_migration():
            cls.parentFunc[str(id(cls)) + func_name] = getattr(
                cls._get_class_gpu_migration(), func_name
            )

        func_list = []
        for orig_module, _ in cls.GLOBAL_FUNC_MAP[id(cls)].items():
            if (
                hasattr(cls, "_get_class_gpu_migration")
                and cls.is_replace_class is False
            ):
                func_list.append(
                    tuple(
                        [
                            cls._get_class_gpu_migration(),
                            orig_module.__name__,
                            cls.wrap_api,
                        ]
                    )
                )
            else:
                func_list.append(
                    tuple([cls.get_module(), orig_module.__name__, cls.wrap_api])
                )

        for child_cls in cls.__subclasses__():
            child_cls.register_cls_func()
            child_cls.register_cls()
            func_list += child_cls.get_functions()
        return func_list

    @classmethod
    def call_parent_func(cls, name, *args, **kwargs):
        func = cls.parentFunc[str(id(cls)) + name]
        if func:
            return func(*args, **kwargs)
        raise ValueError("No {} in parentFunc member".format(name))

    @classmethod
    def get_module(cls) -> ModuleType:
        pass

    @classmethod
    def _save_orig_func_gpu_migration(cls) -> List:
        return []

    @classmethod
    def get_child_all_functions(cls) -> List:
        replace_functions = []
        for child_cls in cls.__subclasses__():
            if child_cls.enabled():
                replace_functions += child_cls.get_functions()

        return replace_functions

    @classmethod
    def register_cls_func(cls):
        """saving class member functions in gpu_migration module and their original version."""
        if cls.is_replace_class:
            return
        for name, func in cls.__dict__.items():
            if hasattr(func, "__name__") and cls.__name__ in str(func) and hasattr(cls, '_get_class_gpu_migration'):
                if hasattr(cls._get_class_gpu_migration(), name):
                    cls.GLOBAL_FUNC_MAP[id(cls)][
                        getattr(cls._get_class_gpu_migration(), name)
                    ] = func

    @classmethod
    def register_cls(cls):
        """saving class in gpu_migration module and their original version."""
        if not cls.is_replace_class:
            return
        cls.GLOBAL_FUNC_MAP[id(cls)][cls._get_class_gpu_migration()] = cls
