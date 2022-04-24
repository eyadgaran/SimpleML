"""
Registry related tests
"""

__author__ = "Elisha Yadgaran"


import unittest
from abc import abstractmethod

import sqlalchemy

from simpleml.persistables.base_sqlalchemy import BaseSQLAlchemy
from simpleml.registries import SIMPLEML_REGISTRY, MetaRegistry, NamedRegistry, Registry
from simpleml.utils.library_versions import safe_lookup


class RegistryTests(unittest.TestCase):
    def test_registry_adds_class_name(self):
        registry = Registry()

        # Define Class
        class FakeClass(object):
            pass

        class_name = "FakeClass"
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
        """
        Test duplicating the same object in the registry doesnt break
        """
        registry = Registry()
        # Define Class

        class FakeClass(object):
            pass

        # Register
        registry.register(FakeClass)
        # Try again - same instantiation
        registry.register(FakeClass)

    def test_getting_missing_key(self):
        fake_key = "blaldfakhfaljaf"
        registry = Registry()
        self.assertEqual(registry.get(fake_key), None)

    def test_context_manager(self):
        registry = Registry()

        # Define Class
        class FakeClass(object):
            pass

        class_name = "FakeClass"
        original_class = FakeClass
        self.assertNotIn(class_name, registry.registry)

        # Register
        registry.register(original_class)

        # Test
        self.assertIn(class_name, registry.registry)
        self.assertEqual(original_class, registry.get(class_name))

        # Different class, same name
        class FakeClass(object):
            pass

        new_class = FakeClass

        self.assertNotEqual(original_class, new_class)

        # overwrite for the duration of the context manager
        with registry.context_register(new_class):
            self.assertIn(class_name, registry.registry)
            self.assertEqual(new_class, registry.get(class_name))

        self.assertIn(class_name, registry.registry)
        self.assertEqual(original_class, registry.get(class_name))

    def test_context_manager_with_error(self):
        registry = Registry()

        # Define Class
        class FakeClass(object):
            pass

        class_name = "FakeClass"
        original_class = FakeClass
        self.assertNotIn(class_name, registry.registry)

        # Register
        registry.register(original_class)

        # Test
        self.assertIn(class_name, registry.registry)
        self.assertEqual(original_class, registry.get(class_name))

        # Different class, same name
        class FakeClass(object):
            pass

        new_class = FakeClass

        self.assertNotEqual(original_class, new_class)

        # overwrite for the duration of the context manager
        with self.assertRaises(ValueError):
            with registry.context_register(new_class):
                self.assertIn(class_name, registry.registry)
                self.assertEqual(new_class, registry.get(class_name))
                raise ValueError()

        self.assertIn(class_name, registry.registry)
        self.assertEqual(original_class, registry.get(class_name))

    def test_context_manager_with_new_key(self):
        registry = Registry()

        # Define Class
        class FakeClass(object):
            pass

        class_name = "FakeClass"
        self.assertNotIn(class_name, registry.registry)

        # overwrite for the duration of the context manager
        with registry.context_register(FakeClass):
            self.assertIn(class_name, registry.registry)
            self.assertEqual(FakeClass, registry.get(class_name))

        self.assertNotIn(class_name, registry.registry)


class NamedRegistryTests(unittest.TestCase):
    def test_registration(self):
        registry = NamedRegistry()

        # Define Class
        class FakeClass(object):
            pass

        name = "test"
        self.assertNotIn(name, registry.registry)

        # Register
        registry.register(name, FakeClass)

        # Test
        self.assertIn(name, registry.registry)
        self.assertEqual(FakeClass, registry.get(name))

    def test_different_key_registration(self):
        """
        Same class, different names
        """
        registry = NamedRegistry()

        # Define Class
        class FakeClass(object):
            pass

        name1 = "test"
        name2 = "test2"
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
        """
        Test duplicate handling for the same class
        """
        registry = NamedRegistry()

        # Define Class
        class FakeClass(object):
            pass

        name = "test"
        self.assertNotIn(name, registry.registry)

        # Register
        registry.register(name, FakeClass)
        self.assertIn(name, registry.registry)
        self.assertEqual(FakeClass, registry.get(name))

        # Test re-registering - should not fail
        registry.register(name, FakeClass, allow_duplicates=True)
        registry.register(name, FakeClass, allow_duplicates=False)

    def test_different_class_duplication(self):
        """
        Test overwrite handling for different classes
        """
        registry = NamedRegistry()

        # Define Class
        class FakeClass(object):
            pass

        # Different class

        class FakeClass2(object):
            pass

        name = "test"
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

    def test_context_manager(self):
        registry = NamedRegistry()

        name = "test"
        original_value = "value"
        self.assertNotIn(name, registry.registry)

        # Register
        registry.register(name, original_value)

        # Test
        self.assertIn(name, registry.registry)
        self.assertEqual(original_value, registry.get(name))

        new_value = "changed"
        self.assertNotEqual(new_value, original_value)

        # overwrite for the duration of the context manager
        with registry.context_register(name, new_value):
            self.assertIn(name, registry.registry)
            self.assertEqual(new_value, registry.get(name))

        self.assertIn(name, registry.registry)
        self.assertEqual(original_value, registry.get(name))

    def test_context_manager_with_error(self):
        registry = NamedRegistry()

        name = "test"
        original_value = "value"
        self.assertNotIn(name, registry.registry)

        # Register
        registry.register(name, original_value)

        # Test
        self.assertIn(name, registry.registry)
        self.assertEqual(original_value, registry.get(name))

        new_value = "changed"
        self.assertNotEqual(new_value, original_value)

        # overwrite for the duration of the context manager
        with self.assertRaises(ValueError):
            with registry.context_register(name, new_value):
                self.assertIn(name, registry.registry)
                self.assertEqual(new_value, registry.get(name))
                raise ValueError()

        self.assertIn(name, registry.registry)
        self.assertEqual(original_value, registry.get(name))

    def test_context_manager_with_new_key(self):
        registry = NamedRegistry()

        name = "test"
        value = "value"
        self.assertNotIn(name, registry.registry)

        # overwrite for the duration of the context manager
        with registry.context_register(name, value):
            self.assertIn(name, registry.registry)
            self.assertEqual(value, registry.get(name))

        self.assertNotIn(name, registry.registry)


class MetaRegistryTests(unittest.TestCase):
    def test_declarative_base_expectation(self):
        """
        sqlalchemy api change to early consume registry
        https://github.com/sqlalchemy/sqlalchemy/blob/a782160de2e66ad6f6cb2630ddc16ced4da1c359/lib/sqlalchemy/orm/decl_api.py#L60
        changed to throwing an error if called without a declarative_base class
        """

        def import_error_class():
            class WILLERRORTEST(metaclass=MetaRegistry):
                __abstract__ = True

            return WILLERRORTEST

        def import_registered_base_class():
            class SHOULDWORKTEST(BaseSQLAlchemy, metaclass=MetaRegistry):
                __abstract__ = True

            return SHOULDWORKTEST

        sqlalchemy_version = safe_lookup("sqlalchemy")
        if sqlalchemy_version > "1.4.0":
            with self.assertRaises(sqlalchemy.exc.InvalidRequestError):
                import_error_class()
        else:
            # should work on older versions
            import_error_class()
        import_registered_base_class()

    def test_abstract_method_error(self):
        class AbstractTestClass(BaseSQLAlchemy, metaclass=MetaRegistry):
            __abstract__ = True

            @abstractmethod
            def blah(self):
                pass

        class FailingTestClass(AbstractTestClass):
            __abstract__ = True
            pass

        with self.assertRaises(TypeError):
            AbstractTestClass()
        with self.assertRaises(TypeError):
            FailingTestClass()

    def test_register_on_import(self):
        def import_new_class():
            class BLAHBLAHTESTCLASS(BaseSQLAlchemy, metaclass=MetaRegistry):
                __abstract__ = True

            return BLAHBLAHTESTCLASS

        class_name = "BLAHBLAHTESTCLASS"
        self.assertNotIn(class_name, SIMPLEML_REGISTRY.registry)

        # Register
        fake_class = import_new_class()

        # Test
        self.assertIn(class_name, SIMPLEML_REGISTRY.registry)
        self.assertEqual(fake_class, SIMPLEML_REGISTRY.get(class_name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
