'''
Persistable related tests
'''

__author__ = 'Elisha Yadgaran'

from simpleml.persistables.base_persistable import BasePersistable
import unittest
from random import randint


class PersistableTests(unittest.TestCase):
    def test_same_class_loads(self):
        # Original Class
        class TestClass(BasePersistable):
            __abstract__ = True

            def _hash(self):
                pass

        cl = TestClass()

        # Change class
        self.assertEqual(cl.__class__, TestClass)
        self.assertNotEqual(cl.__class__, BasePersistable)
        cl.__class__ = BasePersistable
        self.assertEqual(cl.__class__, BasePersistable)
        self.assertNotEqual(cl.__class__, TestClass)

        # See if it reverts on load
        cl.load()
        self.assertEqual(cl.__class__, TestClass)
        self.assertNotEqual(cl.__class__, BasePersistable)

    def test_latest_version_retrieved(self):
        # Original Class
        class VersionTestClass(BasePersistable):
            __tablename__ = 'version_tests'

            def _hash(self):
                return 132435465

        VersionTestClass.__table__.create()

        versions = randint(100, 200)
        for i in range(versions):
            VersionTestClass(name='version_test').save()

        new_class = VersionTestClass(name='version_test')
        self.assertEqual(new_class._get_latest_version(), versions + 1)
