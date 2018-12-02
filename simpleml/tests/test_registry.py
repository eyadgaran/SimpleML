'''
Registry related tests
'''

__author__ = 'Elisha Yadgaran'

from simpleml.persistables.meta_registry import MetaRegistry, Registry, SIMPLEML_REGISTRY
import unittest
from abc import abstractmethod
from future.utils import with_metaclass


class RegistryTests(unittest.TestCase):
    def test_registry_adds_class_name(self):
        registry = Registry()

        # Define Class
        class FakeClass(object):
            pass

        class_name = 'FakeClass'
        self.assertNotIn(class_name, registry.registry)

        # Register
        registry.register(FakeClass)

        # Test
        self.assertIn(class_name, registry.registry)
        self.assertEqual(FakeClass, registry.get(class_name))


    def test_duplicate_class_error(self):
        registry = Registry()

        # Define Class
        class FakeClass(object):
            pass

        class_name = 'FakeClass'
        self.assertNotIn(class_name, registry.registry)

        # Register
        registry.register(FakeClass)
        self.assertIn(class_name, registry.registry)
        self.assertEqual(FakeClass, registry.get(class_name))

        # Try again
        with self.assertRaises(ValueError):
            registry.register(FakeClass)

class MetaRegistryTests(unittest.TestCase):
    def test_abstract_method_error(self):
        class AbstractTestClass(with_metaclass(MetaRegistry, object)):
            __abstract__ = True

            @abstractmethod
            def blah(self):
                pass

        class FailingTestClass(AbstractTestClass):
            pass

        with self.assertRaises(TypeError):
            AbstractTestClass()
        with self.assertRaises(TypeError):
            FailingTestClass()

    def test_register_on_import(self):
        def import_new_class():
            class BLAHBLAHTESTCLASS(with_metaclass(MetaRegistry, object)):
                __abstract__ = True

            return BLAHBLAHTESTCLASS

        class_name = 'BLAHBLAHTESTCLASS'
        self.assertNotIn(class_name, SIMPLEML_REGISTRY.registry)

        # Register
        fake_class = import_new_class()

        # Test
        self.assertIn(class_name, SIMPLEML_REGISTRY.registry)
        self.assertEqual(fake_class, SIMPLEML_REGISTRY.get(class_name))
