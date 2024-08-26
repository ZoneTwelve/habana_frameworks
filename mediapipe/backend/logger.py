import os

log_level = int(os.getenv("MEDIA_PY_LOG_LEVEL", "0"))


def printf(*args, **kwargs):
    if (log_level > 0):
        print(*args, **kwargs)
