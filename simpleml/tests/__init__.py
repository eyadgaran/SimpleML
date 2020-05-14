'''
Combine all the test suites and execute

Testing Paradigm: Hierarchical tests broken out by directory.
Invocation Paths:
    1) python setup.py test
    2) simpleml {test, unit-test, integration-test, regression-test}
    3) python simpleml/tests/[unit, integration, regression, {module-name}]

1) Integrated into setuptools invocation. Done by registering `load_tests` in this
module as the entrypoint for tests
```
    test_suite='simpleml.tests.load_tests'
```

2) Setuptools registered entrypoints resolving to the run_tests functions in the
respective directories. `run_tests` calls `load_tests` so it executes the same tests
as other invokation paths
```
    entry_points = {
        'console_scripts': [
            'simpleml-test=simpleml.tests:run_tests',
            'simpleml-unit-test=simpleml.tests.unit:run_tests',
            'simpleml-integration-test=simpleml.tests.integration:run_tests',
            'simpleml-regression-test=simpleml.tests.regression:run_tests',
        ],
    }
```

3) Calling the modules directly also invokes `run_tests` when calling the __init__.
Otherwise it executes just the tests in the module for easy iteration on specific tests.
```
    if __name__ == '__main__':
        unittest.main()
```
'''

__author__ = 'Elisha Yadgaran'


import unittest

from simpleml.tests.unit import load_tests as unit_test_loader
from simpleml.tests.integration import load_tests as integration_test_loader
from simpleml.tests.regression import load_tests as regression_test_loader


def load_tests(*args, **kwargs):
    unit_tests = unit_test_loader()
    integration_tests = integration_test_loader()
    regression_tests = regression_test_loader()
    all_tests = unittest.TestSuite()
    all_tests.addTests([unit_tests, integration_tests, regression_tests])
    return all_tests


def run_tests():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(load_tests())


if __name__ == '__main__':
    run_tests()
