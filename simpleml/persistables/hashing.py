'''
Mixin classes to handle hashing
'''

__author__ = 'Elisha Yadgaran'


import pandas as pd
import numpy as np
import inspect
import logging

from pandas.util import hash_pandas_object
from typing import Any, Type

from simpleml._external.joblib import hash as deterministic_hash


LOGGER = logging.getLogger(__name__)


class CustomHasherMixin(object):
    '''
    Mixin class to hash any object
    '''
    @classmethod
    def custom_hasher(cls,
                      object_to_hash: Any) -> str:
        """
        Adapted from: https://stackoverflow.com/questions/5884066/hashing-a-dictionary
        Makes a hash from a dictionary, list, tuple or set to any level, that
        contains only other hashable types (including any lists, tuples, sets, and
        dictionaries). In the case where other kinds of objects (like classes) need
        to be hashed, pass in a collection of object attributes that are pertinent.
        For example, a class can be hashed in this fashion:

        custom_hasher([cls.__dict__, cls.__name__])

        A function can be hashed like so:

        custom_hasher([fn.__dict__, fn.__code__])

        python 3.3+ changes the default hash method to add an additional random
        seed. Need to set the global PYTHONHASHSEED=0 or use a different hash
        function
        """
        LOGGER.debug(f'Hashing input: {object_to_hash}')

        # Class attribute dict (mappingproxy class)
        if isinstance(object_to_hash, type(object.__dict__)):
            o2 = {}
            for k, v in object_to_hash.items():
                if not k.startswith("__"):
                    o2[k] = v
            object_to_hash = o2

        LOGGER.debug(f'hash type: {type(object_to_hash)}')

        if isinstance(object_to_hash, (set, tuple, list)):
            hash_output = deterministic_hash(tuple([cls.custom_hasher(e) for e in object_to_hash]))

        elif isinstance(object_to_hash, np.ndarray):
            hash_output = cls.custom_hasher(object_to_hash.tostring())

        elif isinstance(object_to_hash, pd.DataFrame):
            # Pandas is unable to hash numpy arrays so prehash those
            hash_output = hash_pandas_object(object_to_hash.applymap(
                lambda element: cls.custom_hasher(element) if isinstance(element, np.ndarray) else element),
                index=False).sum()

        elif isinstance(object_to_hash, pd.Series):
            # Pandas is unable to hash numpy arrays so prehash those
            hash_output = hash_pandas_object(object_to_hash.apply(
                lambda element: cls.custom_hasher(element) if isinstance(element, np.ndarray) else element),
                index=False).sum()

        elif object_to_hash is None:
            # hash of None is unstable between systems
            hash_output = -12345678987654321

        elif isinstance(object_to_hash, dict):
            hash_output = deterministic_hash(tuple(
                sorted([cls.custom_hasher(item) for item in object_to_hash.items()])
            ))

        elif isinstance(object_to_hash, type(lambda: 0)):
            # Functions dont hash consistently because of the halting problem
            # https://stackoverflow.com/questions/33998594/hash-for-lambda-function-in-python
            # Attempt to use the source code string
            hash_output = cls.custom_hasher(inspect.getsource(object_to_hash))

        elif isinstance(object_to_hash, type):  # uninitialized classes
            # Have to keep this at the end of the try list
            # functions are also of <type 'type'>
            # dynamically defined classes will throw an error trying to find source code
            # also do not want to rigidly define classes as the hash of the code
            # lots of non behavior code changes (extra functionality, comments, etc)
            # can be done and should map to the same "content" (what the hash
            # tries to capture)
            # fall back to just the module.class name instead
            # WARNING: module paths reflect import paths and will be different
            # depending on how a class is imported (from a import cls != from library.a import cls)
            LOGGER.warning(f'Hashing class import path for {object_to_hash}, if a fully qualified import path is not used, calling again from a different location will yield different results!')
            hash_output = cls.custom_hasher(f"{object_to_hash.__module__}.{object_to_hash.__name__}")
            # return self.custom_hasher(inspect.getsource(object_to_hash))

        elif isinstance(object_to_hash, object) and hasattr(object_to_hash, '__dict__'):
            # Everything is an object so keep this at the very end.
            # Should only match initialized objects at this point
            # Represent as a tuple of (class, __dict__)
            hash_output = cls.custom_hasher((object_to_hash.__class__, object_to_hash.__dict__))

        else:
            # primitives (str, int, float)
            # Log a warning if the previous checks were unable to find a suitable
            # decomposition for the object
            if not isinstance(object_to_hash, (float, str, int)):
                LOGGER.warning(f'Unable to find suitable representation of {object_to_hash}, passing through to hash function directly. (This may result in future breaking changes)')
            hash_output = deterministic_hash(object_to_hash)

        LOGGER.debug(f'Hashing output: {hash_output}')
        return hash_output
