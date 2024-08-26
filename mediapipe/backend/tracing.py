import media_pipe_api as mpa


class Singleton(type):
    def __init__(self, name, bases, mmbs):
        super(Singleton, self).__init__(name, bases, mmbs)
        self._instance_ = super(Singleton, self).__call__()

    def __call__(self, *args, **kw):
        return self._instance_


class media_tracer(metaclass=Singleton):
    def __init__(self):
        self.__c_trace = mpa.PyTrace()

    def start_trace(self, name):
        self.__c_trace.StartTrace("py_"+name)

    def end_trace(self, name):
        self.__c_trace.EndTrace("py_"+name)

    def __del__(self):
        del self.__c_trace


class tracer:
    def __init__(self, name):
        self.__c_trace = mpa.PyTrace()
        self.__name = "py_"+name
        self.__c_trace.StartTrace(self.__name)

    def __del__(self):
        self.__c_trace.EndTrace(self.__name)
        del self.__c_trace
