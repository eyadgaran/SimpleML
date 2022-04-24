"""
Tests for save patterns serializers
"""

__author__ = "Elisha Yadgaran"


import pickle
import random
import tempfile
import unittest
from os.path import isfile, join

import cloudpickle
import pandas as pd

from simpleml.imports import dd, ddDataFrame, hickle
from simpleml.save_patterns.serializers.cloudpickle import (
    CloudpickleFileSerializer,
    CloudpickleInMemorySerializer,
    CloudpicklePersistenceMethods,
)
from simpleml.save_patterns.serializers.dask import (
    DaskCSVSerializer,
    DaskJSONSerializer,
    DaskParquetSerializer,
    DaskPersistenceMethods,
)
from simpleml.save_patterns.serializers.hickle import (
    HickleFileSerializer,
    HicklePersistenceMethods,
)
from simpleml.save_patterns.serializers.pandas import (
    PandasCSVSerializer,
    PandasJSONSerializer,
    PandasParquetSerializer,
    PandasPersistenceMethods,
)
from simpleml.save_patterns.serializers.pickle import (
    PickleFileSerializer,
    PickleInMemorySerializer,
    PicklePersistenceMethods,
)

TEMP_DIRECTORY = tempfile.gettempdir()
RANDOM_RUN = random.randint(10000, 99999)


class TestSerializationClass(object):
    """
    Fake test class with all complex datatypes to test pickling
    """

    cls_attribute = "blah"

    def __init__(self, a, *args, **kwargs):
        self.a = a
        self.args = args
        self.kwargs = kwargs

    def __eq__(self, other):
        return all(
            (
                self.cls_attribute == other.cls_attribute,
                self.a == other.a,
                self.args == other.args,
                self.kwargs == other.kwargs,
            )
        )


class CloudPicklePersistenceMethodsTests(unittest.TestCase):
    def test_dump_object(self):
        obj = TestSerializationClass("test_dump_object", 1, 2, 3, other=["abc"])
        filepath = join(
            TEMP_DIRECTORY, f"{self.__class__.__name__}-test_dump_object-{RANDOM_RUN}"
        )
        self.assertFalse(isfile(filepath))
        CloudpicklePersistenceMethods.dump_object(obj, filepath=filepath)
        self.assertTrue(isfile(filepath))

        with open(filepath, "rb") as f:
            deserialized = cloudpickle.load(f)

        self.assertEqual(obj, deserialized)

    def test_dump_object_without_overwrite(self):
        obj = TestSerializationClass(
            "test_dump_object_without_overwrite", 1, 2, 3, other=["abc"]
        )
        obj2 = TestSerializationClass("overwrite", "a", 2, 3, other=["abc"])
        filepath = join(
            TEMP_DIRECTORY,
            f"{self.__class__.__name__}-test_dump_object_without_overwrite-{RANDOM_RUN}",
        )
        CloudpicklePersistenceMethods.dump_object(obj, filepath=filepath)
        self.assertTrue(isfile(filepath))

        # try overwrite
        CloudpicklePersistenceMethods.dump_object(
            obj2, filepath=filepath, overwrite=False
        )

        with open(filepath, "rb") as f:
            deserialized = cloudpickle.load(f)

        self.assertEqual(obj, deserialized)
        self.assertNotEqual(obj2, deserialized)

    def test_dumps_object(self):
        obj = TestSerializationClass("test_dumps_object", 1, 2, 3, other=["abc"])
        serialized = CloudpicklePersistenceMethods.dumps_object(obj)
        deserialized = cloudpickle.loads(serialized)
        self.assertEqual(obj, deserialized)

    def test_load_object(self):
        obj = TestSerializationClass("test_load_object", 1, 2, 3, other=["abc"])
        filepath = join(
            TEMP_DIRECTORY, f"{self.__class__.__name__}-test_load_object-{RANDOM_RUN}"
        )
        self.assertFalse(isfile(filepath))
        with open(filepath, "wb") as f:
            cloudpickle.dump(obj, f)
        self.assertTrue(isfile(filepath))

        deserialized = CloudpicklePersistenceMethods.load_object(filepath)
        self.assertEqual(obj, deserialized)

    def test_loads_object(self):
        obj = TestSerializationClass("test_loads_object", 1, 2, 3, other=["abc"])
        serialized = cloudpickle.dumps(obj)
        deserialized = CloudpicklePersistenceMethods.loads_object(serialized)
        self.assertEqual(obj, deserialized)


class CloudpickleFileSerializerTests(unittest.TestCase):
    def test_serialize(self):
        obj = TestSerializationClass("test_serialize", 1, 2, 3, other=["abc"])
        filepath = f"{self.__class__.__name__}-test_serialize-{RANDOM_RUN}"
        self.assertFalse(isfile(join(TEMP_DIRECTORY, filepath)))
        data = CloudpickleFileSerializer.serialize(
            obj, filepath=filepath, format_directory="", format_extension=""
        )
        self.assertTrue(isfile(join(TEMP_DIRECTORY, filepath)))
        self.assertEqual(data["filepath"], filepath)
        self.assertEqual(data["source_directory"], "system_temp")

        with open(join(TEMP_DIRECTORY, filepath), "rb") as f:
            deserialized = cloudpickle.load(f)

        self.assertEqual(obj, deserialized)

    def test_deserialize(self):
        obj = TestSerializationClass("test_deserialize", 1, 2, 3, other=["abc"])
        filepath = f"{self.__class__.__name__}-test_deserialize-{RANDOM_RUN}"
        self.assertFalse(isfile(join(TEMP_DIRECTORY, filepath)))
        with open(join(TEMP_DIRECTORY, filepath), "wb") as f:
            cloudpickle.dump(obj, f)
        self.assertTrue(isfile(join(TEMP_DIRECTORY, filepath)))
        data = CloudpickleFileSerializer.deserialize(filepath)
        self.assertEqual(obj, data["obj"])


class CloudpickleInMemorySerializerTests(unittest.TestCase):
    def test_serialize(self):
        obj = TestSerializationClass("test_serialize", 1, 2, 3, other=["abc"])
        data = CloudpickleInMemorySerializer.serialize(obj)
        deserialized = cloudpickle.loads(data["obj"])
        self.assertEqual(obj, deserialized)

    def test_deserialize(self):
        obj = TestSerializationClass("test_deserialize", 1, 2, 3, other=["abc"])
        serialized = cloudpickle.dumps(obj)
        data = CloudpickleInMemorySerializer.deserialize(serialized)
        self.assertEqual(obj, data["obj"])


class PicklePersistenceMethodsTests(unittest.TestCase):
    def test_dump_object(self):
        obj = TestSerializationClass("test_dump_object", 1, 2, 3, other=["abc"])
        filepath = join(
            TEMP_DIRECTORY, f"{self.__class__.__name__}-test_dump_object-{RANDOM_RUN}"
        )
        self.assertFalse(isfile(filepath))
        PicklePersistenceMethods.dump_object(obj, filepath=filepath)
        self.assertTrue(isfile(filepath))

        with open(filepath, "rb") as f:
            deserialized = pickle.load(f)

        self.assertEqual(obj, deserialized)

    def test_dump_object_without_overwrite(self):
        obj = TestSerializationClass(
            "test_dump_object_without_overwrite", 1, 2, 3, other=["abc"]
        )
        obj2 = TestSerializationClass("overwrite", "a", 2, 3, other=["abc"])
        filepath = join(
            TEMP_DIRECTORY,
            f"{self.__class__.__name__}-test_dump_object_without_overwrite-{RANDOM_RUN}",
        )
        PicklePersistenceMethods.dump_object(obj, filepath=filepath)
        self.assertTrue(isfile(filepath))

        # try overwrite
        PicklePersistenceMethods.dump_object(obj2, filepath=filepath, overwrite=False)

        with open(filepath, "rb") as f:
            deserialized = pickle.load(f)

        self.assertEqual(obj, deserialized)
        self.assertNotEqual(obj2, deserialized)

    def test_dumps_object(self):
        obj = TestSerializationClass("test_dumps_object", 1, 2, 3, other=["abc"])
        serialized = PicklePersistenceMethods.dumps_object(obj)
        deserialized = pickle.loads(serialized)
        self.assertEqual(obj, deserialized)

    def test_load_object(self):
        obj = TestSerializationClass("test_load_object", 1, 2, 3, other=["abc"])
        filepath = join(
            TEMP_DIRECTORY, f"{self.__class__.__name__}-test_load_object-{RANDOM_RUN}"
        )
        self.assertFalse(isfile(filepath))
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
        self.assertTrue(isfile(filepath))

        deserialized = PicklePersistenceMethods.load_object(filepath)
        self.assertEqual(obj, deserialized)

    def test_loads_object(self):
        obj = TestSerializationClass("test_loads_object", 1, 2, 3, other=["abc"])
        serialized = pickle.dumps(obj)
        deserialized = PicklePersistenceMethods.loads_object(serialized)
        self.assertEqual(obj, deserialized)


class PickleFileSerializerTests(unittest.TestCase):
    def test_serialize(self):
        obj = TestSerializationClass("test_serialize", 1, 2, 3, other=["abc"])
        filepath = f"{self.__class__.__name__}-test_serialize-{RANDOM_RUN}"
        self.assertFalse(isfile(join(TEMP_DIRECTORY, filepath)))
        data = PickleFileSerializer.serialize(
            obj, filepath=filepath, format_directory="", format_extension=""
        )
        self.assertTrue(isfile(join(TEMP_DIRECTORY, filepath)))
        self.assertEqual(data["filepath"], filepath)
        self.assertEqual(data["source_directory"], "system_temp")

        with open(join(TEMP_DIRECTORY, filepath), "rb") as f:
            deserialized = pickle.load(f)

        self.assertEqual(obj, deserialized)

    def test_deserialize(self):
        obj = TestSerializationClass("test_deserialize", 1, 2, 3, other=["abc"])
        filepath = f"{self.__class__.__name__}-test_deserialize-{RANDOM_RUN}"
        self.assertFalse(isfile(join(TEMP_DIRECTORY, filepath)))
        with open(join(TEMP_DIRECTORY, filepath), "wb") as f:
            pickle.dump(obj, f)
        self.assertTrue(isfile(join(TEMP_DIRECTORY, filepath)))
        data = PickleFileSerializer.deserialize(filepath)
        self.assertEqual(obj, data["obj"])


class PickleInMemorySerializerTests(unittest.TestCase):
    def test_serialize(self):
        obj = TestSerializationClass("test_serialize", 1, 2, 3, other=["abc"])
        data = PickleInMemorySerializer.serialize(obj)
        deserialized = pickle.loads(data["obj"])
        self.assertEqual(obj, deserialized)

    def test_deserialize(self):
        obj = TestSerializationClass("test_deserialize", 1, 2, 3, other=["abc"])
        serialized = pickle.dumps(obj)
        data = PickleInMemorySerializer.deserialize(serialized)
        self.assertEqual(obj, data["obj"])


class HicklePersistenceMethodsTests(unittest.TestCase):
    def test_dump_object(self):
        obj = TestSerializationClass("test_dump_object", 1, 2, 3, other=["abc"])
        filepath = join(
            TEMP_DIRECTORY, f"{self.__class__.__name__}-test_dump_object-{RANDOM_RUN}"
        )
        self.assertFalse(isfile(filepath))
        HicklePersistenceMethods.dump_object(obj, filepath=filepath)
        self.assertTrue(isfile(filepath))

        deserialized = hickle.load(filepath)

        self.assertEqual(obj, deserialized)

    def test_dump_object_without_overwrite(self):
        obj = TestSerializationClass(
            "test_dump_object_without_overwrite", 1, 2, 3, other=["abc"]
        )
        obj2 = TestSerializationClass("overwrite", "a", 2, 3, other=["abc"])
        filepath = join(
            TEMP_DIRECTORY,
            f"{self.__class__.__name__}-test_dump_object_without_overwrite-{RANDOM_RUN}",
        )
        HicklePersistenceMethods.dump_object(obj, filepath=filepath)
        self.assertTrue(isfile(filepath))

        # try overwrite
        HicklePersistenceMethods.dump_object(obj2, filepath=filepath, overwrite=False)

        deserialized = hickle.load(filepath)

        self.assertEqual(obj, deserialized)
        self.assertNotEqual(obj2, deserialized)

    def test_load_object(self):
        obj = TestSerializationClass("test_load_object", 1, 2, 3, other=["abc"])
        filepath = join(
            TEMP_DIRECTORY, f"{self.__class__.__name__}-test_load_object-{RANDOM_RUN}"
        )
        self.assertFalse(isfile(filepath))
        hickle.dump(obj, filepath)
        self.assertTrue(isfile(filepath))

        deserialized = HicklePersistenceMethods.load_object(filepath)
        self.assertEqual(obj, deserialized)


class HickleFileSerializerTests(unittest.TestCase):
    def test_serialize(self):
        obj = TestSerializationClass("test_serialize", 1, 2, 3, other=["abc"])
        filepath = f"{self.__class__.__name__}-test_serialize-{RANDOM_RUN}"
        self.assertFalse(isfile(join(TEMP_DIRECTORY, filepath)))
        data = HickleFileSerializer.serialize(
            obj, filepath=filepath, format_directory="", format_extension=""
        )
        self.assertTrue(isfile(join(TEMP_DIRECTORY, filepath)))
        self.assertEqual(data["filepath"], filepath)
        self.assertEqual(data["source_directory"], "system_temp")
        deserialized = hickle.load(join(TEMP_DIRECTORY, filepath))

        self.assertEqual(obj, deserialized)

    def test_deserialize(self):
        obj = TestSerializationClass("test_deserialize", 1, 2, 3, other=["abc"])
        filepath = f"{self.__class__.__name__}-test_deserialize-{RANDOM_RUN}"
        self.assertFalse(isfile(join(TEMP_DIRECTORY, filepath)))
        # with open(join(TEMP_DIRECTORY, filepath), 'wb') as f:
        hickle.dump(obj, join(TEMP_DIRECTORY, filepath))
        self.assertTrue(isfile(join(TEMP_DIRECTORY, filepath)))
        data = HickleFileSerializer.deserialize(filepath)
        self.assertEqual(obj, data["obj"])


class DaskPersistenceMethodsTests(unittest.TestCase):
    def test_read_csv(self):
        pass

    def test_read_parquet(self):
        pass

    def test_read_json(self):
        pass

    def test_to_csv(self):
        pass

    def test_to_parquet(self):
        pass

    def test_to_json(self):
        pass


class DaskParquetSerializerTests(unittest.TestCase):
    def test_serialize(self):
        pass

    def test_deserialize(self):
        pass


class DaskCSVSerializerTests(unittest.TestCase):
    def test_serialize(self):
        pass

    def test_deserialize(self):
        pass


class DaskJSONSerializerTests(unittest.TestCase):
    def test_serialize(self):
        pass

    def test_deserialize(self):
        pass


class PandasPersistenceMethodsTests(unittest.TestCase):
    def test_read_csv(self):
        pass

    def test_read_parquet(self):
        pass

    def test_read_json(self):
        pass

    def test_to_csv(self):
        pass

    def test_to_parquet(self):
        pass

    def test_to_json(self):
        pass


class PandasParquetSerializerTests(unittest.TestCase):
    def test_serialize(self):
        pass

    def test_deserialize(self):
        pass


class PandasCSVSerializerTests(unittest.TestCase):
    def test_serialize(self):
        pass

    def test_deserialize(self):
        pass


class PandasJSONSerializerTests(unittest.TestCase):
    def test_serialize(self):
        pass

    def test_deserialize(self):
        pass


class KerasPersistenceMethodsTests(unittest.TestCase):
    def test_save_model(self):
        pass

    def test_load_model(self):
        pass

    def test_save_weights(self):
        pass

    def test_load_weights(self):
        pass


class KerasSavedModelSerializerTests(unittest.TestCase):
    def test_serialize(self):
        pass

    def test_deserialize(self):
        pass


class KerasH5SerializerTests(unittest.TestCase):
    def test_serialize(self):
        pass

    def test_deserialize(self):
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
