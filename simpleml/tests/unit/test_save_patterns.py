"""
Tests for base save pattern functionality
"""

__author__ = "Elisha Yadgaran"


import unittest

from simpleml.registries import LOAD_METHOD_REGISTRY, SAVE_METHOD_REGISTRY
from simpleml.save_patterns.base import BaseSavePattern, BaseSerializer
from simpleml.save_patterns.decorators import (
    SavePatternDecorators,
    deregister_save_pattern,
    register_save_pattern,
)


class SavePatternRegistrationTests(unittest.TestCase):
    """
    Test for registration decorators and functions
    """

    def test_registering_new_save_pattern_with_decorator_explicitly(self):
        """
        Decorator test with save pattern parameter
        """
        save_pattern = "fake_explicit_decorated_save_pattern"
        SAVE_METHOD_REGISTRY.drop(save_pattern)
        LOAD_METHOD_REGISTRY.drop(save_pattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        @SavePatternDecorators.register_save_pattern(
            save_pattern, save=True, load=False
        )
        class FakeSavePattern(object):
            pass

        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertEqual(SAVE_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

    def test_registering_new_save_pattern_with_decorator_implicitly(self):
        """
        Decorator test with class attribute for save pattern
        """
        save_pattern = "fake_implicit_decorated_save_pattern"
        SAVE_METHOD_REGISTRY.drop(save_pattern)
        LOAD_METHOD_REGISTRY.drop(save_pattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        @SavePatternDecorators.register_save_pattern(save=True, load=False)
        class FakeSavePattern(object):
            SAVE_PATTERN = save_pattern
            pass

        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertEqual(SAVE_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

    def test_registering_new_load_pattern_with_decorator_explicitly(self):
        """
        Decorator test with load pattern parameter
        """
        save_pattern = "fake_explicit_decorated_load_pattern"
        SAVE_METHOD_REGISTRY.drop(save_pattern)
        LOAD_METHOD_REGISTRY.drop(save_pattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        @SavePatternDecorators.register_save_pattern(
            save_pattern, save=False, load=True
        )
        class FakeSavePattern(object):
            pass

        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)
        self.assertEqual(LOAD_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)

    def test_registering_new_load_pattern_with_decorator_implicitly(self):
        """
        Decorator test with class attribute for load pattern
        """
        save_pattern = "fake_implicit_decorated_load_pattern"
        SAVE_METHOD_REGISTRY.drop(save_pattern)
        LOAD_METHOD_REGISTRY.drop(save_pattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        @SavePatternDecorators.register_save_pattern(save=False, load=True)
        class FakeSavePattern(object):
            SAVE_PATTERN = save_pattern
            pass

        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)
        self.assertEqual(LOAD_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)

    def test_registering_both_patterns_with_decorator_explicitly(self):
        """
        Decorator test with pattern parameter
        """
        save_pattern = "fake_explicit_decorated_pattern"
        SAVE_METHOD_REGISTRY.drop(save_pattern)
        LOAD_METHOD_REGISTRY.drop(save_pattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        @SavePatternDecorators.register_save_pattern(save_pattern, save=True, load=True)
        class FakeSavePattern(object):
            pass

        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertEqual(SAVE_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)
        self.assertEqual(LOAD_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)

    def test_registering_both_patterns_with_decorator_implicitly(self):
        """
        Decorator test without pattern parameter
        """
        save_pattern = "fake_implicit_decorated_pattern"
        SAVE_METHOD_REGISTRY.drop(save_pattern)
        LOAD_METHOD_REGISTRY.drop(save_pattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        @SavePatternDecorators.register_save_pattern(save=True, load=True)
        class FakeSavePattern(object):
            SAVE_PATTERN = save_pattern

        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertEqual(SAVE_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)
        self.assertEqual(LOAD_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)

    def test_registering_with_decorator_without_parameters(self):
        save_pattern = "fake_decorated_without_parameters"
        SAVE_METHOD_REGISTRY.drop(save_pattern)
        LOAD_METHOD_REGISTRY.drop(save_pattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        @SavePatternDecorators.register_save_pattern
        class FakeSavePattern(object):
            SAVE_PATTERN = save_pattern

        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertEqual(SAVE_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)
        self.assertEqual(LOAD_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)

    def test_registering_new_save_pattern_explicitly(self):
        """
        test with save pattern parameter
        """
        save_pattern = "fake_explicit_save_pattern"
        SAVE_METHOD_REGISTRY.drop(save_pattern)
        LOAD_METHOD_REGISTRY.drop(save_pattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        class FakeSavePattern(object):
            pass

        register_save_pattern(
            FakeSavePattern, save_pattern=save_pattern, save=True, load=False
        )

        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertEqual(SAVE_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

    def test_registering_new_save_pattern_implicitly(self):
        """
        test with class attribute for save pattern
        """
        save_pattern = "fake_implicit_save_pattern"
        SAVE_METHOD_REGISTRY.drop(save_pattern)
        LOAD_METHOD_REGISTRY.drop(save_pattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        class FakeSavePattern(object):
            SAVE_PATTERN = save_pattern

        register_save_pattern(FakeSavePattern, save=True, load=False)

        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertEqual(SAVE_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

    def test_registering_new_load_pattern_explicitly(self):
        """
        test with load pattern parameter
        """
        save_pattern = "fake_explicit_load_pattern"
        SAVE_METHOD_REGISTRY.drop(save_pattern)
        LOAD_METHOD_REGISTRY.drop(save_pattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        class FakeSavePattern(object):
            pass

        register_save_pattern(
            FakeSavePattern, save_pattern=save_pattern, save=False, load=True
        )

        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)
        self.assertEqual(LOAD_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)

    def test_registering_new_load_pattern_implicitly(self):
        """
        test with class attribute for load pattern
        """
        save_pattern = "fake_implicit_load_pattern"
        SAVE_METHOD_REGISTRY.drop(save_pattern)
        LOAD_METHOD_REGISTRY.drop(save_pattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        class FakeSavePattern(object):
            SAVE_PATTERN = save_pattern

        register_save_pattern(FakeSavePattern, save=False, load=True)

        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)
        self.assertEqual(LOAD_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)

    def test_registering_both_patterns_explicitly(self):
        """
        test with pattern parameter
        """
        save_pattern = "fake_explicit_pattern"
        SAVE_METHOD_REGISTRY.drop(save_pattern)
        LOAD_METHOD_REGISTRY.drop(save_pattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        class FakeSavePattern(object):
            pass

        register_save_pattern(
            FakeSavePattern, save_pattern=save_pattern, save=True, load=True
        )

        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertEqual(SAVE_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)
        self.assertEqual(LOAD_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)

    def test_registering_both_patterns_implicitly(self):
        """
        test without pattern parameter
        """
        save_pattern = "fake_implicit_pattern"
        SAVE_METHOD_REGISTRY.drop(save_pattern)
        LOAD_METHOD_REGISTRY.drop(save_pattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        class FakeSavePattern(object):
            SAVE_PATTERN = save_pattern

        register_save_pattern(FakeSavePattern, save=True, load=True)

        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertEqual(SAVE_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)
        self.assertEqual(LOAD_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)

    def test_registering_without_parameters(self):
        save_pattern = "fake_without_parameters"
        SAVE_METHOD_REGISTRY.drop(save_pattern)
        LOAD_METHOD_REGISTRY.drop(save_pattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        class FakeSavePattern(object):
            SAVE_PATTERN = save_pattern

        register_save_pattern(FakeSavePattern)

        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertEqual(SAVE_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)
        self.assertEqual(LOAD_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)

    def test_deregistering_new_save_pattern_with_decorator_explicitly(self):
        """
        Decorator test with save pattern parameter
        """
        save_pattern = "fake_explicit_decorated_save_pattern"
        SAVE_METHOD_REGISTRY.registry[save_pattern] = None
        LOAD_METHOD_REGISTRY.registry[save_pattern] = None
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        @SavePatternDecorators.deregister_save_pattern(
            save_pattern, save=True, load=False
        )
        class FakeSavePattern(object):
            pass

        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

    def test_deregistering_new_save_pattern_with_decorator_implicitly(self):
        """
        Decorator test with class attribute for save pattern
        """
        save_pattern = "fake_implicit_decorated_save_pattern"
        SAVE_METHOD_REGISTRY.registry[save_pattern] = None
        LOAD_METHOD_REGISTRY.registry[save_pattern] = None
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        @SavePatternDecorators.deregister_save_pattern(save=True, load=False)
        class FakeSavePattern(object):
            SAVE_PATTERN = save_pattern

        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

    def test_deregistering_new_load_pattern_with_decorator_explicitly(self):
        """
        Decorator test with load pattern parameter
        """
        save_pattern = "fake_explicit_decorated_load_pattern"
        SAVE_METHOD_REGISTRY.registry[save_pattern] = None
        LOAD_METHOD_REGISTRY.registry[save_pattern] = None
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        @SavePatternDecorators.deregister_save_pattern(
            save_pattern, save=False, load=True
        )
        class FakeSavePattern(object):
            pass

        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)

    def test_deregistering_new_load_pattern_with_decorator_implicitly(self):
        """
        Decorator test with class attribute for load pattern
        """
        save_pattern = "fake_implicit_decorated_load_pattern"
        SAVE_METHOD_REGISTRY.registry[save_pattern] = None
        LOAD_METHOD_REGISTRY.registry[save_pattern] = None
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        @SavePatternDecorators.deregister_save_pattern(save=False, load=True)
        class FakeSavePattern(object):
            SAVE_PATTERN = save_pattern

        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)

    def test_deregistering_both_patterns_with_decorator_explicitly(self):
        """
        Decorator test with pattern parameter
        """
        save_pattern = "fake_explicit_decorated_pattern"
        SAVE_METHOD_REGISTRY.registry[save_pattern] = None
        LOAD_METHOD_REGISTRY.registry[save_pattern] = None
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        @SavePatternDecorators.deregister_save_pattern(
            save_pattern, save=True, load=True
        )
        class FakeSavePattern(object):
            pass

        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

    def test_deregistering_both_patterns_with_decorator_implicitly(self):
        """
        Decorator test without pattern parameter
        """
        save_pattern = "fake_implicit_decorated_pattern"
        SAVE_METHOD_REGISTRY.registry[save_pattern] = None
        LOAD_METHOD_REGISTRY.registry[save_pattern] = None
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        @SavePatternDecorators.deregister_save_pattern(save=True, load=True)
        class FakeSavePattern(object):
            SAVE_PATTERN = save_pattern

        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

    def test_deregistering_with_decorator_without_parameters(self):
        save_pattern = "fake_decorated_without_parameters"
        SAVE_METHOD_REGISTRY.registry[save_pattern] = None
        LOAD_METHOD_REGISTRY.registry[save_pattern] = None
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        @SavePatternDecorators.deregister_save_pattern
        class FakeSavePattern(object):
            SAVE_PATTERN = save_pattern

        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

    def test_deregistering_new_save_pattern_explicitly(self):
        """
        test with save pattern parameter
        """
        save_pattern = "fake_explicit_save_pattern"
        SAVE_METHOD_REGISTRY.registry[save_pattern] = None
        LOAD_METHOD_REGISTRY.registry[save_pattern] = None
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        class FakeSavePattern(object):
            pass

        deregister_save_pattern(
            FakeSavePattern, save_pattern=save_pattern, save=True, load=False
        )

        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

    def test_deregistering_new_save_pattern_implicitly(self):
        """
        test with class attribute for save pattern
        """
        save_pattern = "fake_implicit_save_pattern"
        SAVE_METHOD_REGISTRY.registry[save_pattern] = None
        LOAD_METHOD_REGISTRY.registry[save_pattern] = None
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        class FakeSavePattern(object):
            SAVE_PATTERN = save_pattern

        deregister_save_pattern(FakeSavePattern, save=True, load=False)

        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

    def test_deregistering_new_load_pattern_explicitly(self):
        """
        test with load pattern parameter
        """
        save_pattern = "fake_explicit_load_pattern"
        SAVE_METHOD_REGISTRY.registry[save_pattern] = None
        LOAD_METHOD_REGISTRY.registry[save_pattern] = None
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        class FakeSavePattern(object):
            pass

        deregister_save_pattern(
            FakeSavePattern, save_pattern=save_pattern, save=False, load=True
        )

        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)

    def test_deregistering_new_load_pattern_implicitly(self):
        """
        test with class attribute for load pattern
        """
        save_pattern = "fake_implicit_load_pattern"
        SAVE_METHOD_REGISTRY.registry[save_pattern] = None
        LOAD_METHOD_REGISTRY.registry[save_pattern] = None
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        class FakeSavePattern(object):
            SAVE_PATTERN = save_pattern

        deregister_save_pattern(FakeSavePattern, save=False, load=True)

        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)

    def test_deregistering_both_patterns_explicitly(self):
        """
        test with pattern parameter
        """
        save_pattern = "fake_explicit_pattern"
        SAVE_METHOD_REGISTRY.registry[save_pattern] = None
        LOAD_METHOD_REGISTRY.registry[save_pattern] = None
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        class FakeSavePattern(object):
            pass

        deregister_save_pattern(
            FakeSavePattern, save_pattern=save_pattern, save=True, load=True
        )

        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

    def test_deregistering_both_patterns_implicitly(self):
        """
        test without pattern parameter
        """
        save_pattern = "fake_implicit_pattern"
        SAVE_METHOD_REGISTRY.registry[save_pattern] = None
        LOAD_METHOD_REGISTRY.registry[save_pattern] = None
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        class FakeSavePattern(object):
            SAVE_PATTERN = save_pattern

        deregister_save_pattern(FakeSavePattern, save=True, load=True)

        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

    def test_deregistering_without_parameters(self):
        save_pattern = "fake_without_parameters"
        SAVE_METHOD_REGISTRY.registry[save_pattern] = None
        LOAD_METHOD_REGISTRY.registry[save_pattern] = None
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        class FakeSavePattern(object):
            SAVE_PATTERN = save_pattern

        deregister_save_pattern(FakeSavePattern)

        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)


class BaseSavePatternTests(unittest.TestCase):
    """
    Tests functionality of base save pattern wrapper
    """

    pass


class BaseSerializerTests(unittest.TestCase):
    """
    Tests functionality of abstract base serializer class
    """

    pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
