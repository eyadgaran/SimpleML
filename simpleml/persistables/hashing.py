'''
Mixin classes to handle hashing
'''

__author__ = 'Elisha Yadgaran'


import copy
import pandas as pd
from pandas.util import hash_pandas_object


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
            return tuple([self.custom_hasher(e) for e in object_to_hash])

        elif isinstance(object_to_hash, (pd.DataFrame, pd.Series)):
            return hash_pandas_object(object_to_hash, index=False).sum()

        elif object_to_hash is None:
            # hash of None is unstable between systems
            return -12345678987654321

        elif not isinstance(object_to_hash, dict):
            return hash(object_to_hash)

        new_object_to_hash = copy.deepcopy(object_to_hash)
        for k, v in new_object_to_hash.items():
            new_object_to_hash[k] = self.custom_hasher(v)

        return hash(tuple(frozenset(sorted(new_object_to_hash.items()))))
