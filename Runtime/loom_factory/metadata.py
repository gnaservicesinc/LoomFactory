"""Darwin file metadata access without an additional Python package."""

import ctypes
import errno
import json
import os
from pathlib import Path

ATTRIBUTE = "com.gnaservicesinc.loomfactory.image.v1"


def read_image_metadata(path):
    path = os.fsencode(path)
    if hasattr(os, "getxattr"):
        try:
            data = os.getxattr(path, ATTRIBUTE)
        except OSError as error:
            if error.errno in (errno.ENODATA, 93):
                return None
            raise
    else:
        library = ctypes.CDLL(None, use_errno=True)
        read = library.getxattr
        read.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
                         ctypes.c_size_t, ctypes.c_uint32, ctypes.c_int]
        read.restype = ctypes.c_ssize_t
        size = read(path, ATTRIBUTE.encode(), None, 0, 0, 0)
        if size < 0:
            error = ctypes.get_errno()
            if error == 93:  # ENOATTR on Darwin
                return None
            raise OSError(error, os.strerror(error), os.fsdecode(path))
        if not 0 < size < 1024 * 1024:
            raise ValueError(f"Image metadata could not be read: {Path(os.fsdecode(path)).name}")
        buffer = ctypes.create_string_buffer(size)
        count = read(path, ATTRIBUTE.encode(), buffer, size, 0, 0)
        if count != size:
            raise ValueError("Image metadata changed while being read. Refresh the source folder.")
        data = buffer.raw
    return json.loads(data)


def write_fixture_metadata(path, value):
    """For new validation fixtures only; production image editing belongs to Swift."""
    data = json.dumps(value).encode()
    if hasattr(os, "setxattr"):
        os.setxattr(path, ATTRIBUTE, data)
        return
    library = ctypes.CDLL(None, use_errno=True)
    write = library.setxattr
    write.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
                      ctypes.c_size_t, ctypes.c_uint32, ctypes.c_int]
    write.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(data)
    if write(os.fsencode(path), ATTRIBUTE.encode(), buffer, len(data), 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(path))
