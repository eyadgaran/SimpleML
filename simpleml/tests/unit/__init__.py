'''
Unit tests are limited to functional tests that do not require a
database connection
'''

__author__ = 'Elisha Yadgaran'


import glob
import os
import unittest

from coverage import Coverage

DIRECTORY_IMPORT_PATH = 'simpleml.tests.unit'
DIRECTORY_ABSOLUTE_PATH = os.path.dirname(__file__)


class UnitTestSuite(unittest.TestSuite):
    def __init__(self):
        # Load all tests in the directory
        loader = unittest.TestLoader()
        module_paths = glob.glob(os.path.join(DIRECTORY_ABSOLUTE_PATH, 'test*.py'), recursive=True)  # /root/abspath/tests/test*.py
        if len(DIRECTORY_ABSOLUTE_PATH) == 0:  # launched from the local directory - no prefix to strip out
            relative_module_paths = [i[len(DIRECTORY_ABSOLUTE_PATH):] for i in module_paths]  # find the delta from this filepath test*.py
        else:
            relative_module_paths = [i[len(DIRECTORY_ABSOLUTE_PATH) + 1:] for i in module_paths]  # find the delta from this filepath test*.py
        module_imports = ['.'.join((DIRECTORY_IMPORT_PATH, i.replace('/', '.')[:-3])) for i in relative_module_paths]  # Import string simpleml.tests.test*
        tests = [loader.loadTestsFromName(i) for i in module_imports]
        super().__init__(tests=tests)

    def setUp(self):
        print('Running Unit Tests')

    def tearDown(self):
        print('Finished Running Unit Tests')

    def run(self, *args, **kwargs):
        self.setUp()
        super().run(*args, **kwargs)
        self.tearDown()


def load_tests(*args, **kwargs):
    return UnitTestSuite()


def run_tests():
    # Start coverage collection
    cov = Coverage(
        context='unit',
        data_file='.coverage.unit'
    )
    cov.start()

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(load_tests())

    # Stop and save
    cov.stop()
    cov.save()

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)


if __name__ == '__main__':
    run_tests()
