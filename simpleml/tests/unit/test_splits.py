'''
Split related tests
'''

__author__ = 'Elisha Yadgaran'


import unittest

import pandas as pd

from simpleml.datasets.dataset_splits import Split, SplitContainer


class DatasetSplitTests(unittest.TestCase):
    def test_null_type_check(self):
        for null_type in (
            None,
            pd.DataFrame(),
            pd.Series(),
            [],
            tuple(),
            {}
        ):
            self.assertTrue(Split.is_null_type(null_type))

    def test_squeeze_behavior(self):
        split = Split(a=None, b=pd.DataFrame(), c=pd.Series(), d=[], e=tuple(), f={})
        split.squeeze()
        self.assertEqual(split, {})

    def test_getattr_behavior(self):
        split = Split(a='ab')
        self.assertEqual(split.a, 'ab')
        self.assertEqual(split.b, None)


class DatasetSplitContainerTests(unittest.TestCase):
    def test_default_value(self):
        container = SplitContainer()
        self.assertEqual(container['nonexistent_key'], Split())


if __name__ == '__main__':
    unittest.main(verbosity=2)
