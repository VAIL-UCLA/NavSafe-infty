"""General utilities"""
import logging
from functools import wraps
from time import time

# Library logging must not configure the application root logger or create
# files in the caller's working directory merely by being imported.
logger = logging.getLogger("navsafe.pysocialforce")
logger.addHandler(logging.NullHandler())


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.debug(f"Timeit: {f.__name__}({args}, {kw}), took: {te-ts:2.4f} sec")
        return result

    return wrap
