'''
Tests for save patterns serializers
'''

__author__ = 'Elisha Yadgaran'


import random
import tempfile
import unittest
from os.path import isfile, join

TEMP_DIRECTORY = tempfile.gettempdir()
RANDOM_RUN = random.randint(10000, 99999)


class TestSerializationClass(object):
    '''
    Fake test class with all complex datatypes to test pickling
    '''
    cls_attribute = 'blah'

    def __init__(self, a, *args, **kwargs):
        self.a = a
        self.args = args
        self.kwargs = kwargs

    def __eq__(self, other):
        return all((
            self.cls_attribute == other.cls_attribute,
            self.a == other.a,
            self.args == other.args,
            self.kwargs == other.kwargs
        ))


class CloudpickleFileSerializerTests(unittest.TestCase):
    def test_deserialize_serialize_output(self):
        pass


class CloudpickleInMemorySerializerTests(unittest.TestCase):
    def test_deserialize_serialize_output(self):
        pass


class PickleFileSerializerTests(unittest.TestCase):
    def test_deserialize_serialize_output(self):
        pass


class PickleInMemorySerializerTests(unittest.TestCase):
    def test_deserialize_serialize_output(self):
        pass


class HickleFileSerializerTests(unittest.TestCase):
    def test_deserialize_serialize_output(self):
        pass


class PandasParquetSerializerTests(unittest.TestCase):
    def test_deserialize_serialize_output(self):
        pass


class PandasCSVSerializerTests(unittest.TestCase):
    def test_deserialize_serialize_output(self):
        pass


class PandasJSONSerializerTests(unittest.TestCase):
    def test_deserialize_serialize_output(self):
        pass


class DaskParquetSerializerTests(unittest.TestCase):
    def test_deserialize_serialize_output(self):
        pass


class DaskCSVSerializerTests(unittest.TestCase):
    def test_deserialize_serialize_output(self):
        pass


class DaskJSONSerializerTests(unittest.TestCase):
    def test_deserialize_serialize_output(self):
        pass


class KerasSavedModelSerializerTests(unittest.TestCase):
    def test_deserialize_serialize_output(self):
        pass


class KerasH5SerializerTests(unittest.TestCase):
    def test_deserialize_serialize_output(self):
        pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
