'''
Registry related tests
'''

__author__ = 'Elisha Yadgaran'

from simpleml.registries import MetaRegistry, Registry, SIMPLEML_REGISTRY, NamedRegistry
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

        # Register
        registry.register(FakeClass)

        # Different class, same name
        class FakeClass(object):
            pass
        with self.assertRaises(ValueError):
            registry.register(FakeClass)

    def test_reloaded_class_registers(self):
        '''
        Test duplicating the same object in the registry doesnt break
        '''
        registry = Registry()
        # Define Class

        class FakeClass(object):
            pass

        # Register
        registry.register(FakeClass)
        # Try again - same instantiation
        registry.register(FakeClass)

    def test_getting_missing_key(self):
        fake_key = 'blaldfakhfaljaf'
        registry = Registry()
        self.assertEqual(registry.get(fake_key), None)


class NamedRegistryTests(unittest.TestCase):
    def test_registration(self):
        registry = NamedRegistry()

        # Define Class
        class FakeClass(object):
            pass

        name = 'test'
        self.assertNotIn(name, registry.registry)

        # Register
        registry.register(name, FakeClass)

        # Test
        self.assertIn(name, registry.registry)
        self.assertEqual(FakeClass, registry.get(name))

    def test_different_key_registration(self):
        '''
        Same class, different names
        '''
        registry = NamedRegistry()

        # Define Class
        class FakeClass(object):
            pass

        name1 = 'test'
        name2 = 'test2'
        self.assertNotIn(name1, registry.registry)
        self.assertNotIn(name2, registry.registry)

        # Register
        registry.register(name1, FakeClass)
        registry.register(name2, FakeClass)

        # Test
        self.assertIn(name1, registry.registry)
        self.assertIn(name2, registry.registry)
        self.assertEqual(FakeClass, registry.get(name1))
        self.assertEqual(FakeClass, registry.get(name2))

    def test_same_class_duplication(self):
        '''
        Test duplicate handling for the same class
        '''
        registry = NamedRegistry()

        # Define Class
        class FakeClass(object):
            pass

        name = 'test'
        self.assertNotIn(name, registry.registry)

        # Register
        registry.register(name, FakeClass)
        self.assertIn(name, registry.registry)
        self.assertEqual(FakeClass, registry.get(name))

        # Test re-registering - should not fail
        registry.register(name, FakeClass, allow_duplicates=True)
        registry.register(name, FakeClass, allow_duplicates=False)

    def test_different_class_duplication(self):
        '''
        Test overwrite handling for different classes
        '''
        registry = NamedRegistry()

        # Define Class
        class FakeClass(object):
            pass
        # Different class

        class FakeClass2(object):
            pass

        name = 'test'
        self.assertNotIn(name, registry.registry)

        # Register
        registry.register(name, FakeClass)
        self.assertIn(name, registry.registry)
        self.assertEqual(FakeClass, registry.get(name))

        # Test re-registering
        with self.assertRaises(ValueError):
            registry.register(name, FakeClass2, allow_duplicates=False)
        self.assertEqual(FakeClass, registry.get(name))

        registry.register(name, FakeClass2, allow_duplicates=True)
        self.assertEqual(FakeClass2, registry.get(name))


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


if __name__ == '__main__':
    unittest.main(verbosity=2)
