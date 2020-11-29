'''
Test Suites that validate changes against historical benchmarks
and artifacts
'''

__author__ = 'Elisha Yadgaran'


import unittest
import os
import glob


DIRECTORY_IMPORT_PATH = 'simpleml.tests.regression'
DIRECTORY_ABSOLUTE_PATH = os.path.dirname(__file__)


class RegressionTestSuite(unittest.TestSuite):
    def __init__(self):
        # Load all tests in the directory
        loader = unittest.TestLoader()
        module_paths = glob.glob(os.path.join(DIRECTORY_ABSOLUTE_PATH, 'test*.py'), recursive=True)  # /root/abspath/tests/test*.py
        relative_module_paths = [i[len(DIRECTORY_ABSOLUTE_PATH) + 1:] for i in module_paths]  # find the delta from this filepath test*.py
        module_imports = ['.'.join((DIRECTORY_IMPORT_PATH, i.replace('/', '.')[:-3])) for i in relative_module_paths]  # Import string simpleml.tests.test*
        tests = [loader.loadTestsFromName(i) for i in module_imports]
        super().__init__(tests=tests)

    def setUp(self):
        print('Running Regression Tests')

    def tearDown(self):
        print('Finished Running Regression Tests')

    def run(self, *args, **kwargs):
        self.setUp()
        super().run(*args, **kwargs)
        self.tearDown()


def load_tests(*args, **kwargs):
    return RegressionTestSuite()


def run_tests():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(load_tests())


if __name__ == '__main__':
    run_tests()
