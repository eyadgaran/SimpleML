'''
Persistable related tests
'''

__author__ = 'Elisha Yadgaran'

from simpleml.persistables.base_persistable import Persistable
import unittest


class PersistableTests(unittest.TestCase):
    def test_same_class_loads(self):
        # Original Class
        class TestClass(Persistable):
            __abstract__ = True

            def _hash(self):
                pass

        cl = TestClass()

        # Change class
        self.assertEqual(cl.__class__, TestClass)
        self.assertNotEqual(cl.__class__, Persistable)
        cl.__class__ = Persistable
        self.assertEqual(cl.__class__, Persistable)
        self.assertNotEqual(cl.__class__, TestClass)

        # See if it reverts on load
        cl.load()
        self.assertEqual(cl.__class__, TestClass)
        self.assertNotEqual(cl.__class__, Persistable)

    def test_lazy_loading(self):
        '''
        Test that dependecy loads only when called
        '''
        # Delete dependency file and check that it works
        # Then reference dependency and assert error

    def test_abstract_hash_error(self):
        '''
        Confirm an error is raised if initialized without
        defining the hash
        '''

    def test_loading_without_externals(self):
        pass

    def test_loading_with_externals(self):
        pass

    def test_class_loading(self):
        pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
