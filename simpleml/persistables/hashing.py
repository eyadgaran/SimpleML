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

        python 3.3+ changes the default hash method to add an additional random
        seed. Need to set the global PYTHONHASHSEED=0 or use a different hash
        function
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


"""
Copied from joblib library -- no modification, just avoiding unnecessary dependencies
https://github.com/joblib/joblib/blob/master/joblib/hashing.py#L246

Fast cryptographic hash of Python objects, with a special case for fast
hashing of numpy arrays.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import pickle
import hashlib
import sys
import types
import struct
import io
import decimal

PY3_OR_LATER = sys.version_info[0] >= 3
try:
    _basestring = basestring
    _bytes_or_unicode = (str, unicode)
except NameError:
    _basestring = str
    _bytes_or_unicode = (bytes, str)


if PY3_OR_LATER:
    Pickler = pickle._Pickler
else:
    Pickler = pickle.Pickler


class _ConsistentSet(object):
    """ Class used to ensure the hash of Sets is preserved
        whatever the order of its items.
    """
    def __init__(self, set_sequence):
        # Forces order of elements in set to ensure consistent hash.
        try:
            # Trying first to order the set assuming the type of elements is
            # consistent and orderable.
            # This fails on python 3 when elements are unorderable
            # but we keep it in a try as it's faster.
            self._sequence = sorted(set_sequence)
        except (TypeError, decimal.InvalidOperation):
            # If elements are unorderable, sorting them using their hash.
            # This is slower but works in any case.
            self._sequence = sorted((hash(e) for e in set_sequence))


class _MyHash(object):
    """ Class used to hash objects that won't normally pickle """

    def __init__(self, *args):
        self.args = args


class Hasher(Pickler):
    """ A subclass of pickler, to do cryptographic hashing, rather than
        pickling.
    """

    def __init__(self, hash_name='md5'):
        self.stream = io.BytesIO()
        # By default we want a pickle protocol that only changes with
        # the major python version and not the minor one
        protocol = (pickle.DEFAULT_PROTOCOL if PY3_OR_LATER
                    else pickle.HIGHEST_PROTOCOL)
        Pickler.__init__(self, self.stream, protocol=protocol)
        # Initialise the hash obj
        self._hash = hashlib.new(hash_name)

    def hash(self, obj, return_digest=True):
        try:
            self.dump(obj)
        except pickle.PicklingError as e:
            e.args += ('PicklingError while hashing %r: %r' % (obj, e),)
            raise
        dumps = self.stream.getvalue()
        self._hash.update(dumps)
        if return_digest:
            return self._hash.hexdigest()

    def save(self, obj):
        if isinstance(obj, (types.MethodType, type({}.pop))):
            # the Pickler cannot pickle instance methods; here we decompose
            # them into components that make them uniquely identifiable
            if hasattr(obj, '__func__'):
                func_name = obj.__func__.__name__
            else:
                func_name = obj.__name__
            inst = obj.__self__
            if type(inst) == type(pickle):
                obj = _MyHash(func_name, inst.__name__)
            elif inst is None:
                # type(None) or type(module) do not pickle
                obj = _MyHash(func_name, inst)
            else:
                cls = obj.__self__.__class__
                obj = _MyHash(func_name, inst, cls)
        Pickler.save(self, obj)

    def memoize(self, obj):
        # We want hashing to be sensitive to value instead of reference.
        # For example we want ['aa', 'aa'] and ['aa', 'aaZ'[:2]]
        # to hash to the same value and that's why we disable memoization
        # for strings
        if isinstance(obj, _bytes_or_unicode):
            return
        Pickler.memoize(self, obj)

    # The dispatch table of the pickler is not accessible in Python
    # 3, as these lines are only bugware for IPython, we skip them.
    def save_global(self, obj, name=None, pack=struct.pack):
        # We have to override this method in order to deal with objects
        # defined interactively in IPython that are not injected in
        # __main__
        kwargs = dict(name=name, pack=pack)
        if sys.version_info >= (3, 4):
            del kwargs['pack']
        try:
            Pickler.save_global(self, obj, **kwargs)
        except pickle.PicklingError:
            Pickler.save_global(self, obj, **kwargs)
            module = getattr(obj, "__module__", None)
            if module == '__main__':
                my_name = name
                if my_name is None:
                    my_name = obj.__name__
                mod = sys.modules[module]
                if not hasattr(mod, my_name):
                    # IPython doesn't inject the variables define
                    # interactively in __main__
                    setattr(mod, my_name, obj)

    dispatch = Pickler.dispatch.copy()
    # builtin
    dispatch[type(len)] = save_global
    # type
    dispatch[type(object)] = save_global
    # classobj
    dispatch[type(Pickler)] = save_global
    # function
    dispatch[type(pickle.dump)] = save_global

    def _batch_setitems(self, items):
        # forces order of keys in dict to ensure consistent hash.
        try:
            # Trying first to compare dict assuming the type of keys is
            # consistent and orderable.
            # This fails on python 3 when keys are unorderable
            # but we keep it in a try as it's faster.
            Pickler._batch_setitems(self, iter(sorted(items)))
        except TypeError:
            # If keys are unorderable, sorting them using their hash. This is
            # slower but works in any case.
            Pickler._batch_setitems(self, iter(sorted((hash(k), v)
                                                      for k, v in items)))

    def save_set(self, set_items):
        # forces order of items in Set to ensure consistent hash
        Pickler.save(self, _ConsistentSet(set_items))

    dispatch[type(set())] = save_set


class NumpyHasher(Hasher):
    """ Special case the hasher for when numpy is loaded.
    """

    def __init__(self, hash_name='md5', coerce_mmap=False):
        """
            Parameters
            ----------
            hash_name: string
                The hash algorithm to be used
            coerce_mmap: boolean
                Make no difference between np.memmap and np.ndarray
                objects.
        """
        self.coerce_mmap = coerce_mmap
        Hasher.__init__(self, hash_name=hash_name)
        # delayed import of numpy, to avoid tight coupling
        import numpy as np
        self.np = np
        if hasattr(np, 'getbuffer'):
            self._getbuffer = np.getbuffer
        else:
            self._getbuffer = memoryview

    def save(self, obj):
        """ Subclass the save method, to hash ndarray subclass, rather
            than pickling them. Off course, this is a total abuse of
            the Pickler class.
        """
        if isinstance(obj, self.np.ndarray) and not obj.dtype.hasobject:
            # Compute a hash of the object
            # The update function of the hash requires a c_contiguous buffer.
            if obj.shape == ():
                # 0d arrays need to be flattened because viewing them as bytes
                # raises a ValueError exception.
                obj_c_contiguous = obj.flatten()
            elif obj.flags.c_contiguous:
                obj_c_contiguous = obj
            elif obj.flags.f_contiguous:
                obj_c_contiguous = obj.T
            else:
                # Cater for non-single-segment arrays: this creates a
                # copy, and thus aleviates this issue.
                # XXX: There might be a more efficient way of doing this
                obj_c_contiguous = obj.flatten()

            # memoryview is not supported for some dtypes, e.g. datetime64, see
            # https://github.com/numpy/numpy/issues/4983. The
            # workaround is to view the array as bytes before
            # taking the memoryview.
            self._hash.update(
                self._getbuffer(obj_c_contiguous.view(self.np.uint8)))

            # We store the class, to be able to distinguish between
            # Objects with the same binary content, but different
            # classes.
            if self.coerce_mmap and isinstance(obj, self.np.memmap):
                # We don't make the difference between memmap and
                # normal ndarrays, to be able to reload previously
                # computed results with memmap.
                klass = self.np.ndarray
            else:
                klass = obj.__class__
            # We also return the dtype and the shape, to distinguish
            # different views on the same data with different dtypes.

            # The object will be pickled by the pickler hashed at the end.
            obj = (klass, ('HASHED', obj.dtype, obj.shape, obj.strides))
        elif isinstance(obj, self.np.dtype):
            # Atomic dtype objects are interned by their default constructor:
            # np.dtype('f8') is np.dtype('f8')
            # This interning is not maintained by a
            # pickle.loads + pickle.dumps cycle, because __reduce__
            # uses copy=True in the dtype constructor. This
            # non-deterministic behavior causes the internal memoizer
            # of the hasher to generate different hash values
            # depending on the history of the dtype object.
            # To prevent the hash from being sensitive to this, we use
            # .descr which is a full (and never interned) description of
            # the array dtype according to the numpy doc.
            klass = obj.__class__
            obj = (klass, ('HASHED', obj.descr))
        Hasher.save(self, obj)


def hash(obj, hash_name='md5', coerce_mmap=False):
    """ Quick calculation of a hash to identify uniquely Python objects
        containing numpy arrays.
        Parameters
        -----------
        hash_name: 'md5' or 'sha1'
            Hashing algorithm used. sha1 is supposedly safer, but md5 is
            faster.
        coerce_mmap: boolean
            Make no difference between np.memmap and np.ndarray
    """
    if 'numpy' in sys.modules:
        hasher = NumpyHasher(hash_name=hash_name, coerce_mmap=coerce_mmap)
    else:
        hasher = Hasher(hash_name=hash_name)
    return hasher.hash(obj)
