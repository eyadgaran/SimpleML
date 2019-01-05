'''
Mixin classes to handle hashing
'''

__author__ = 'Elisha Yadgaran'


import pandas as pd
import numpy as np
from pandas.util import hash_pandas_object
import inspect


class CustomHasherMixin(object):
    '''
    Mixin class to hash any object
    '''
    def custom_hasher(self, object_to_hash, custom_class_proxy=type(object.__dict__)):
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
        """
        if type(object_to_hash) == custom_class_proxy:
            o2 = {}
            for k, v in object_to_hash.items():
                if not k.startswith("__"):
                    o2[k] = v
            object_to_hash = o2

        if isinstance(object_to_hash, (set, tuple, list)):
            return hash(tuple([self.custom_hasher(e) for e in object_to_hash]))

        elif isinstance(object_to_hash, np.ndarray):
            return self.custom_hasher(object_to_hash.tostring())

        elif isinstance(object_to_hash, pd.DataFrame):
            # Pandas is unable to hash numpy arrays so prehash those
            return hash_pandas_object(object_to_hash.applymap(
                lambda element: self.custom_hasher(element) if isinstance(element, np.ndarray) else element),
                index=False).sum()

        elif isinstance(object_to_hash, pd.Series):
            # Pandas is unable to hash numpy arrays so prehash those
            return hash_pandas_object(object_to_hash.apply(
                lambda element: self.custom_hasher(element) if isinstance(element, np.ndarray) else element),
                index=False).sum()

        elif object_to_hash is None:
            # hash of None is unstable between systems
            return -12345678987654321

        elif isinstance(object_to_hash, dict):
            return hash(tuple(
                sorted([self.custom_hasher(item) for item in object_to_hash.items()])
            ))

        elif isinstance(object_to_hash, type(lambda: 0)):
            # Functions dont hash consistently because of the halting problem
            # https://stackoverflow.com/questions/33998594/hash-for-lambda-function-in-python
            # Attempt to use the source code string
            return self.custom_hasher(inspect.getsource(object_to_hash))

        elif isinstance(object_to_hash, type):
            # Have to keep this at the end of the try list; np.ndarray,
            # pd.DataFrame/Series, and function are also of <type 'type'>
            return self.custom_hasher(repr(object_to_hash))
            # return self.custom_hasher(inspect.getsource(object_to_hash))

        else:
            return hash(object_to_hash)
