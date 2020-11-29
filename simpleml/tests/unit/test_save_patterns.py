'''
Tests for save patterns
'''

__author__ = 'Elisha Yadgaran'


import unittest
import tempfile
import random

from os.path import isfile, join

from simpleml.registries import SAVE_METHOD_REGISTRY, LOAD_METHOD_REGISTRY
from simpleml.save_patterns.decorators import SavePatternDecorators, register_save_pattern, deregister_save_pattern
from simpleml.save_patterns.base import SavePatternMixin, BaseSavePattern
from simpleml.save_patterns.database import DatabaseTableSavePattern, DatabasePickleSavePattern
from simpleml.save_patterns.libcloud import CloudBase, CloudPickleSavePattern, CloudHDF5SavePattern, CloudKerasHDF5SavePattern
from simpleml.save_patterns.local import DiskPickleSavePattern, DiskHDF5SavePattern
from simpleml.save_patterns.onedrive import OnedriveBase, OnedrivePickleSavePattern, OnedriveHDF5SavePattern, OnedriveKerasHDF5SavePattern


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


class SavePatternRegistrationTests(unittest.TestCase):
    '''
    Test for registration decorators and functions
    '''

    def test_registering_new_save_pattern_with_decorator_explicitly(self):
        '''
        Decorator test with save pattern parameter
        '''
        save_pattern = 'fake_explicit_decorated_save_pattern'
        SAVE_METHOD_REGISTRY.drop(save_pattern)
        LOAD_METHOD_REGISTRY.drop(save_pattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        @SavePatternDecorators.register_save_pattern(save_pattern, save=True, load=False)
        class FakeSavePattern(object):
            pass

        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertEqual(SAVE_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

    def test_registering_new_save_pattern_with_decorator_implicitly(self):
        '''
        Decorator test with class attribute for save pattern
        '''
        save_pattern = 'fake_implicit_decorated_save_pattern'
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
        '''
        Decorator test with load pattern parameter
        '''
        save_pattern = 'fake_explicit_decorated_load_pattern'
        SAVE_METHOD_REGISTRY.drop(save_pattern)
        LOAD_METHOD_REGISTRY.drop(save_pattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        @SavePatternDecorators.register_save_pattern(save_pattern, save=False, load=True)
        class FakeSavePattern(object):
            pass

        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)
        self.assertEqual(LOAD_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)

    def test_registering_new_load_pattern_with_decorator_implicitly(self):
        '''
        Decorator test with class attribute for load pattern
        '''
        save_pattern = 'fake_implicit_decorated_load_pattern'
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
        '''
        Decorator test with pattern parameter
        '''
        save_pattern = 'fake_explicit_decorated_pattern'
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
        '''
        Decorator test without pattern parameter
        '''
        save_pattern = 'fake_implicit_decorated_pattern'
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
        save_pattern = 'fake_decorated_without_parameters'
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
        '''
        test with save pattern parameter
        '''
        save_pattern = 'fake_explicit_save_pattern'
        SAVE_METHOD_REGISTRY.drop(save_pattern)
        LOAD_METHOD_REGISTRY.drop(save_pattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        class FakeSavePattern(object):
            pass

        register_save_pattern(FakeSavePattern, save_pattern=save_pattern, save=True, load=False)

        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertEqual(SAVE_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

    def test_registering_new_save_pattern_implicitly(self):
        '''
        test with class attribute for save pattern
        '''
        save_pattern = 'fake_implicit_save_pattern'
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
        '''
        test with load pattern parameter
        '''
        save_pattern = 'fake_explicit_load_pattern'
        SAVE_METHOD_REGISTRY.drop(save_pattern)
        LOAD_METHOD_REGISTRY.drop(save_pattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        class FakeSavePattern(object):
            pass

        register_save_pattern(FakeSavePattern, save_pattern=save_pattern, save=False, load=True)

        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)
        self.assertEqual(LOAD_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)

    def test_registering_new_load_pattern_implicitly(self):
        '''
        test with class attribute for load pattern
        '''
        save_pattern = 'fake_implicit_load_pattern'
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
        '''
        test with pattern parameter
        '''
        save_pattern = 'fake_explicit_pattern'
        SAVE_METHOD_REGISTRY.drop(save_pattern)
        LOAD_METHOD_REGISTRY.drop(save_pattern)
        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        class FakeSavePattern(object):
            pass

        register_save_pattern(FakeSavePattern, save_pattern=save_pattern, save=True, load=True)

        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertEqual(SAVE_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)
        self.assertEqual(LOAD_METHOD_REGISTRY.get(save_pattern), FakeSavePattern)

    def test_registering_both_patterns_implicitly(self):
        '''
        test without pattern parameter
        '''
        save_pattern = 'fake_implicit_pattern'
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
        save_pattern = 'fake_without_parameters'
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
        '''
        Decorator test with save pattern parameter
        '''
        save_pattern = 'fake_explicit_decorated_save_pattern'
        SAVE_METHOD_REGISTRY.registry[save_pattern] = None
        LOAD_METHOD_REGISTRY.registry[save_pattern] = None
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        @SavePatternDecorators.deregister_save_pattern(save_pattern, save=True, load=False)
        class FakeSavePattern(object):
            pass

        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

    def test_deregistering_new_save_pattern_with_decorator_implicitly(self):
        '''
        Decorator test with class attribute for save pattern
        '''
        save_pattern = 'fake_implicit_decorated_save_pattern'
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
        '''
        Decorator test with load pattern parameter
        '''
        save_pattern = 'fake_explicit_decorated_load_pattern'
        SAVE_METHOD_REGISTRY.registry[save_pattern] = None
        LOAD_METHOD_REGISTRY.registry[save_pattern] = None
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        @SavePatternDecorators.deregister_save_pattern(save_pattern, save=False, load=True)
        class FakeSavePattern(object):
            pass

        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)

    def test_deregistering_new_load_pattern_with_decorator_implicitly(self):
        '''
        Decorator test with class attribute for load pattern
        '''
        save_pattern = 'fake_implicit_decorated_load_pattern'
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
        '''
        Decorator test with pattern parameter
        '''
        save_pattern = 'fake_explicit_decorated_pattern'
        SAVE_METHOD_REGISTRY.registry[save_pattern] = None
        LOAD_METHOD_REGISTRY.registry[save_pattern] = None
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        @SavePatternDecorators.deregister_save_pattern(save_pattern, save=True, load=True)
        class FakeSavePattern(object):
            pass

        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

    def test_deregistering_both_patterns_with_decorator_implicitly(self):
        '''
        Decorator test without pattern parameter
        '''
        save_pattern = 'fake_implicit_decorated_pattern'
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
        save_pattern = 'fake_decorated_without_parameters'
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
        '''
        test with save pattern parameter
        '''
        save_pattern = 'fake_explicit_save_pattern'
        SAVE_METHOD_REGISTRY.registry[save_pattern] = None
        LOAD_METHOD_REGISTRY.registry[save_pattern] = None
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        class FakeSavePattern(object):
            pass

        deregister_save_pattern(FakeSavePattern, save_pattern=save_pattern, save=True, load=False)

        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

    def test_deregistering_new_save_pattern_implicitly(self):
        '''
        test with class attribute for save pattern
        '''
        save_pattern = 'fake_implicit_save_pattern'
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
        '''
        test with load pattern parameter
        '''
        save_pattern = 'fake_explicit_load_pattern'
        SAVE_METHOD_REGISTRY.registry[save_pattern] = None
        LOAD_METHOD_REGISTRY.registry[save_pattern] = None
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        class FakeSavePattern(object):
            pass

        deregister_save_pattern(FakeSavePattern, save_pattern=save_pattern, save=False, load=True)

        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)

    def test_deregistering_new_load_pattern_implicitly(self):
        '''
        test with class attribute for load pattern
        '''
        save_pattern = 'fake_implicit_load_pattern'
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
        '''
        test with pattern parameter
        '''
        save_pattern = 'fake_explicit_pattern'
        SAVE_METHOD_REGISTRY.registry[save_pattern] = None
        LOAD_METHOD_REGISTRY.registry[save_pattern] = None
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        class FakeSavePattern(object):
            pass

        deregister_save_pattern(FakeSavePattern, save_pattern=save_pattern, save=True, load=True)

        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

    def test_deregistering_both_patterns_implicitly(self):
        '''
        test without pattern parameter
        '''
        save_pattern = 'fake_implicit_pattern'
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
        save_pattern = 'fake_without_parameters'
        SAVE_METHOD_REGISTRY.registry[save_pattern] = None
        LOAD_METHOD_REGISTRY.registry[save_pattern] = None
        self.assertIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertIn(save_pattern, LOAD_METHOD_REGISTRY.registry)

        class FakeSavePattern(object):
            SAVE_PATTERN = save_pattern

        deregister_save_pattern(FakeSavePattern)

        self.assertNotIn(save_pattern, SAVE_METHOD_REGISTRY.registry)
        self.assertNotIn(save_pattern, LOAD_METHOD_REGISTRY.registry)


class SavePatternTests(unittest.TestCase, SavePatternMixin):
    '''
    Unit tests for save pattern behavior
    '''

    def test_pickling_in_memory(self):
        '''
        Asserts pickle stream unpickles to the same object
        '''
        obj = TestSerializationClass('pickle')
        stream = self.pickle_object(obj)
        self.assertTrue(isinstance(stream, bytes))
        deserialized = self.load_pickled_object(stream, stream=True)
        self.assertEqual(obj, deserialized)

    def test_pickling_to_disk(self):
        '''
        Asserts pickling and unpickling are the same object and that
        the expected filepath is written to
        '''
        obj = TestSerializationClass('pickle')
        filepath = f'pickle_unit_test-{RANDOM_RUN}'
        self.assertFalse(isfile(join(TEMP_DIRECTORY, filepath)))
        self.pickle_object(obj, filepath=filepath, root_directory=TEMP_DIRECTORY)
        self.assertTrue(isfile(join(TEMP_DIRECTORY, filepath)))
        deserialized = self.load_pickled_object(filepath, root_directory=TEMP_DIRECTORY)
        self.assertEqual(obj, deserialized)

    def test_hickling_to_disk(self):
        '''
        Asserts pickling and unpickling are the same object and that
        the expected filepath is written to
        '''
        obj = TestSerializationClass('hickle')
        filepath = f'hickle_unit_test-{RANDOM_RUN}'
        self.assertFalse(isfile(join(TEMP_DIRECTORY, filepath)))
        self.hickle_object(obj, filepath=filepath, root_directory=TEMP_DIRECTORY)
        self.assertTrue(isfile(join(TEMP_DIRECTORY, filepath)))
        deserialized = self.load_hickled_object(filepath, root_directory=TEMP_DIRECTORY)
        self.assertEqual(obj, deserialized)

    def test_pickling_to_disk_with_overwrite(self):
        '''
        Asserts overwrite functionality for pickle
        '''
        obj = TestSerializationClass('pickle_dummy_original')
        obj2 = TestSerializationClass('pickle_dummy_overwritten')
        self.assertNotEqual(obj, obj2)
        filepath = f'pickle_unit_test_overwrite-{RANDOM_RUN}'
        self.assertFalse(isfile(join(TEMP_DIRECTORY, filepath)))
        self.pickle_object(obj, filepath=filepath, root_directory=TEMP_DIRECTORY)
        self.assertTrue(isfile(join(TEMP_DIRECTORY, filepath)))

        # Attempt to overwrite
        self.pickle_object(obj2, filepath=filepath, root_directory=TEMP_DIRECTORY, overwrite=False)
        self.assertTrue(isfile(join(TEMP_DIRECTORY, filepath)))
        deserialized = self.load_pickled_object(filepath, root_directory=TEMP_DIRECTORY)
        self.assertEqual(obj, deserialized)

        self.pickle_object(obj2, filepath=filepath, root_directory=TEMP_DIRECTORY, overwrite=True)
        self.assertTrue(isfile(join(TEMP_DIRECTORY, filepath)))
        deserialized = self.load_pickled_object(filepath, root_directory=TEMP_DIRECTORY)
        self.assertEqual(obj2, deserialized)

    def test_hickling_to_disk_with_overwrite(self):
        '''
        Asserts overwrite functionality for hickle
        '''
        obj = TestSerializationClass('hickle_dummy_original')
        obj2 = TestSerializationClass('hickle_dummy_overwritten')
        self.assertNotEqual(obj, obj2)
        filepath = f'hickle_unit_test_overwrite-{RANDOM_RUN}'
        self.assertFalse(isfile(join(TEMP_DIRECTORY, filepath)))
        self.hickle_object(obj, filepath=filepath, root_directory=TEMP_DIRECTORY)
        self.assertTrue(isfile(join(TEMP_DIRECTORY, filepath)))

        # Attempt to overwrite
        self.hickle_object(obj2, filepath=filepath, root_directory=TEMP_DIRECTORY, overwrite=False)
        self.assertTrue(isfile(join(TEMP_DIRECTORY, filepath)))
        deserialized = self.load_hickled_object(filepath, root_directory=TEMP_DIRECTORY)
        self.assertEqual(obj, deserialized)

        self.hickle_object(obj2, filepath=filepath, root_directory=TEMP_DIRECTORY, overwrite=True)
        self.assertTrue(isfile(join(TEMP_DIRECTORY, filepath)))
        deserialized = self.load_hickled_object(filepath, root_directory=TEMP_DIRECTORY)
        self.assertEqual(obj2, deserialized)

    @unittest.skip('Requires active postgres connection')
    def test_inserting_dataframe_into_database(self):
        '''
        Test inserting dataframe into a postgres database.
        Assumes postgres connection available and configured
        '''


if __name__ == '__main__':
    unittest.main(verbosity=2)
