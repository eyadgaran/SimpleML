'''
Test import handling
'''

__author__ = 'Elisha Yadgaran'


import unittest
from simpleml.imports import MissingImportFactory


class ImportHandlingTests(unittest.TestCase):
    @property
    def missing_import(self):
        return MissingImportFactory('test', 'test', 'testing')

    def test_missing_import_used_as_base_class(self):
        missing_import = self.missing_import

        with self.subTest(missing_import=missing_import, msg='subclass without init'):
            class DependentSubclass(missing_import):
                pass
            with self.assertRaises(ImportError):
                DependentSubclass()

        with self.subTest(missing_import=missing_import, msg='subclass with init dead end'):
            class DependentSubclassWithDeadEndInit(missing_import):
                def __init__(self):
                    pass
            with self.assertRaises(ImportError):
                DependentSubclassWithDeadEndInit()

        with self.subTest(missing_import=missing_import, msg='subclass with forwarding init'):
            class DependentSubclassWithForwardingInit(missing_import):
                def __init__(self):
                    pass
            with self.assertRaises(ImportError):
                DependentSubclassWithForwardingInit()

    def test_missing_import_used_as_callable(self):
        with self.assertRaises(ImportError):
            self.missing_import()


if __name__ == '__main__':
    unittest.main(verbosity=2)
