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

import enum
import importlib
import inspect
import os
from datetime import datetime

from habana_frameworks.torch.gpu_migration.core import _utils


class LogMode(enum.IntEnum):
    """
    Specifies how messages should be logged.
    """

    EACH = 0
    """Log the message each time"""
    ONCE = 1
    """Log the message only once. The same message will not be logged again."""


APITYPE = [
    "hpu_match",
    "hpu_modified",
    "hpu_mismatch",
]


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class LogArgs:
    def __init__(
        self,
        prefix: str = "",
        call_func: str = "",
        old_call: dict = None,
        new_call: str = "",
        api_type: str = "",
    ) -> None:
        assert api_type in APITYPE, "{} is not in {}".format(api_type, APITYPE)
        self.api_type = api_type
        self.call_func = call_func
        if old_call is None:
            self.old_call = {}
        else:
            self.old_call = old_call
        self.new_call = new_call
        self.prefix_name = prefix

    def __str__(self) -> str:
        api_name = "{}.{}".format(self.prefix_name, self.call_func)
        user_call = ""
        if isinstance(self.old_call, dict):
            if "self" in self.old_call.keys():
                del self.old_call["self"]

            user_call = _utils.kwargs_to_str(self.old_call)

        return "[{}]: {}({}) --> {}".format(
            self.api_type, api_name, user_call, self.new_call
        )

    def __eq__(self, __o: object) -> bool:
        return str(self) == str(__o)

    def __hash__(self) -> int:
        return hash(str(self))


class Logger:
    """
    Global logging interface.
    """

    EXTRA_VERBOSE = 0
    """Enable extra verbose messages and above"""
    VERBOSE = 10
    """Enable verbose messages and above"""
    INFO = 20
    """Enable informative messages and above"""
    DISABLE = 30

    COLOR_MAPPING = {
        "hpu_match": bcolors.OKGREEN,
        "hpu_modified": bcolors.WARNING,
        "hpu_mismatch": bcolors.OKBLUE,
    }

    def __init__(
        self,
        severity=DISABLE,
        colors=True,
        letter=True,
        timestamp=False,
        line_info=True,
    ):
        self.logging_indent = 4
        self.once_logged = set()
        self.colors = colors
        self.letter = letter
        self.timestamp = timestamp
        self.line_info = line_info
        self._log_file = None
        self.module_severity = severity

    def find_all_files(self, base):
        for root, _, fs in os.walk(base):
            for f in fs:
                yield os.path.join(root, f)

    @property
    def module_severity(self):
        return self._module_severity

    @module_severity.setter
    def module_severity(self, value):
        self._module_severity = value

    @property
    def log_file(self):
        return self._log_path

    @log_file.setter
    def log_file(self, value=None):
        if value is None:
            return

        habana_log_dir = os.getenv("HABANA_LOGS")
        if habana_log_dir is not None:
            base_dir = os.path.join(habana_log_dir, "gpu_migration_logs")
        else:
            base_dir = os.path.join(os.getcwd(), "gpu_migration_logs")
        format_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S").split(" ")
        dir_path = os.path.join(base_dir, *format_time)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        self._log_path = os.path.join(dir_path, value)
        print("gpu migration log will be saved to {}".format(self._log_path))
        self._log_file = open(self._log_path, "w")

        exclude_modules = (
            ["habana_frameworks.torch.gpu_migration", "torch"]
            if self.module_severity >= Logger.VERBOSE
            else ["habana_frameworks.torch.gpu_migration"]
        )
        self.all_exclude_files = []
        for exclude in exclude_modules:
            find_dirs = importlib.util.find_spec(exclude).submodule_search_locations
            for find_dir in find_dirs:
                self.all_exclude_files.append(find_dir)

    def log(self, func_name, message, mode=LogMode.ONCE):
        def should_exclude(f):
            for d in self.all_exclude_files:
                if len(d) <= len(f) and f[: len(d)] == d:
                    return True
            return False

        def get_call_file_line_and_context(call_stack):
            return call_stack.filename, call_stack.lineno, call_stack.code_context

        def get_ref_file_path_and_lineo():
            stacks = inspect.stack()
            call_file = None
            call_line = None
            call_context = None

            for i, obj in enumerate(stacks):
                if obj.function == func_name:
                    call_stack = stacks[i + 1]
                    call_file, call_line, call_context = get_call_file_line_and_context(call_stack)
                    if should_exclude(call_file):
                        del stacks
                        return None, None, None
            del stacks
            return call_file, call_line, call_context

        def process_message(file_path, line_no, context):
            message_line = "[{}] {}:{}\n".format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file_path, line_no
            )

            message_line += "{}[context]: {}".format(
                " " * self.logging_indent, " ".join(context)
            )
            return message_line

        file_path, line_no, context = get_ref_file_path_and_lineo()

        def should_log(file_path):
            if file_path is None:
                return False

            if mode == LogMode.EACH:
                return True

            if not tuple([file_path, line_no, message]) in self.once_logged:
                self.once_logged.add(tuple([file_path, line_no, message]))
                return True
            else:
                return False

        if not should_log(file_path):
            return

        log_message = process_message(file_path, line_no, context)

        description = "{}{}\n".format(" " * self.logging_indent, message)

        if self._log_file is not None:
            if isinstance(log_message, str):
                self._log_file.write(log_message)
                self._log_file.write(description)
            elif log_message is None:
                self._log_file.write(description)
            self._log_file.write("\n")
            self._log_file.flush()

        if self.module_severity < Logger.VERBOSE:
            if log_message is not None:
                print(log_message)

            if self.colors:
                print(self.color + description + bcolors.ENDC)
            else:
                print(description)

    def info(
        self,
        func_prefix: str = "",
        api_type: str = "",
        old_args: dict = None,
        new_call: str = "",
        mode=LogMode.ONCE,
        stack=1,
    ):
        """_summary_

        Args:
            func_prefix (str, optional): _description_. Defaults to "".
            api_type (str, optional): _description_. Defaults to "".
            old_args (dict, optional): the function arguments of interface user call. Defaults to {}.
            new_call (str, optional): _description_. Defaults to "".
            mode (_type_, optional): _description_. Defaults to LogMode.ONCE.
        """
        if old_args is None:
            old_args = {}
        if self.module_severity <= Logger.INFO:
            func_name = inspect.stack()[stack][3]
            format_message = LogArgs(
                api_type=api_type,
                prefix=func_prefix,
                call_func=func_name,
                old_call=old_args,
                new_call=new_call,
            )

            self.color = self.COLOR_MAPPING[api_type]
            self.log(func_name, format_message, mode=mode)


G_LOGGER = Logger()
"""The global logger. Use this instead of constructing a logger"""
