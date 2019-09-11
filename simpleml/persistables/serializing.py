'''
Module to define the classes that support serialization for all JSON columns

A lot of persistable subclasses try to persist objects that cannot be natively serialized
in a database. This attempts to define a custom pattern to serialize them and
subsequently deserialize on load
'''

__author__ = 'Elisha Yadgaran'


import simplejson as json
import cloudpickle as pickle
import codecs
import sys
from decimal import Decimal
from datetime import datetime, timedelta

try:  # Py2/3 compatibility
    basestring
except NameError:
    basestring = str


class JSONSerializer(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.__str__()
        if isinstance(obj, timedelta):
            return int(obj)
        # Otherwise try defaults or fallback to pickle
        try:
            return super(JSONSerializer, self).default(obj)
        except TypeError:
            return 'pickle_serialized->' + codecs.encode(pickle.dumps(obj), "base64").decode()


def object_hook(data, ignore_dicts=False):
    # if this is a string, check for pickled
    if isinstance(data, basestring) and len(data) > 19:
        if data[:19] == 'pickle_serialized->':
            return pickle.loads(codecs.decode(data[19:].encode(), "base64"))

    # if this is a list of values, return list of deserialized values
    if isinstance(data, list):
        return [object_hook(item, ignore_dicts=True) for item in data]

    # if this is a dictionary, return dictionary of deserialized keys and values
    # but only if we haven't already deserialized it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            object_hook(key, ignore_dicts=True): object_hook(value, ignore_dicts=True)
            for key, value in data.items()
        }

    # if it's anything else, return it in its original form
    return data


def custom_dumps(data):
    # Unfortunately JSON doesnt support dict keys as ints so they will automatically
    # get converted. Hopefully this wont be an issue in SimpleML, but be aware...
    # Ignore NaN converts any non-native json types (NaN, Infinity, etc) to null. Native json library
    # does not handle those and doesnt recognize them as unserializable (so overwriting default doesnt help)
    return json.dumps(data, cls=JSONSerializer, ensure_ascii=False, ignore_nan=True)


def custom_loads(data):
    # Need the extra nesting because json loads only passes nested dict objects to object_hook
    # Python 2 strings have a decode method, python 3 doesn't
    if sys.version_info[0] == 2:
        return object_hook(json.loads(data.decode('utf-8'), object_hook=object_hook), ignore_dicts=True)
    else:
        return object_hook(json.loads(data, object_hook=object_hook), ignore_dicts=True)
