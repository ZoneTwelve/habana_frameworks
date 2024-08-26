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

import importlib
import pathlib
import sys
from collections import namedtuple
from typing import List, Tuple

_func_entry = namedtuple("_func_entry", "orig_module func_name func_wrap")


class ModuleReplacementHelper:
    """To help replace CUDA replated APIs with HPU APIs (by monkey patching)"""

    def __init__(self):
        self.visited = set()
        self.root_path = pathlib.Path(__file__).parent.parent.parent

    def prepare_state(self, functions=None):
        """
        Prepare the function map that would used in the monkey patching mechanism later. To avoid processing an API more than once, use a set to record visited APIs.
        """
        func_map = set()

        if functions is not None:
            for item in functions:
                name = ".".join([item[0].__name__, item[1]])
                if name not in self.visited:
                    self.visited.add(name)
                else:
                    continue
                func_map.add(_func_entry(item[0], item[1], item[2]))
        return func_map

    def apply_func_wrapper(self, func_map):
        for entry in func_map:
            func = getattr(entry.orig_module, entry.func_name)
            setattr(entry.orig_module, entry.func_name, entry.func_wrap(func))

    def initialize(self, modules_func: List[Tuple] = None):
        func_map = self.prepare_state(modules_func)
        self.apply_func_wrapper(func_map)

    def do_import(self, name: str):
        # Put the module in gpu_migration available
        if name in sys.modules:
            _ = sys.modules.pop(name, None)

        module = importlib.import_module(
            "habana_frameworks.torch.gpu_migration." + name
        )
        sys.modules[name] = module


function_helper = ModuleReplacementHelper()
