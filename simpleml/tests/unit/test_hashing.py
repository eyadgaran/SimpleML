'''
Hashing related tests
'''

__author__ = 'Elisha Yadgaran'


import unittest
import pandas as pd

from simpleml.persistables.hashing import CustomHasherMixin
from simpleml._external.joblib import hash as deterministic_hash


class _Test123(object):
    random_attribute = 'abc'

    def __init__(self):
        pass

    def fancy_method(self):
        pass

    def __repr__(self):
        return 'not actually used in hashing'


class CustomHasherTests(unittest.TestCase):
    '''
    Hashing tests for consistency across environment and machines.
    Expectations generated on Mac running python 3.7

    Tests trace recursive behavior via log assertions
    '''

    def test_initialized_class_hashing(self):
        '''
        Hashes the initialized object directly (via pickle)
        '''

        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            expected_final_hash = 'e4728209b833849314b0f05dc59c6096'
            with self.subTest():
                self.assertEqual(CustomHasherMixin.custom_hasher(_Test123()), expected_final_hash)

            # internal behavior
            # hash object
            self.maxDiff = None
            self.assertEqual(
                logs.output,
                [f"DEBUG:simpleml.persistables.hashing:Hashing input: {_Test123()}",
                 f"DEBUG:simpleml.persistables.hashing:hash type: {_Test123}",
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}"])

    def test_uninitialized_class_hashing(self):
        '''
        Hashes the repr(cls) for initialized objects
        '''

        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            expected_final_hash = '0392beff85f9eb48de4cf51f2b7ee139'
            with self.subTest():
                self.assertEqual(CustomHasherMixin.custom_hasher(_Test123), expected_final_hash)

            # internal behavior
            # hash class -> hash class repr
            self.maxDiff = None
            self.assertEqual(
                logs.output,
                [f"DEBUG:simpleml.persistables.hashing:Hashing input: {_Test123}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'type'>",
                 f"DEBUG:simpleml.persistables.hashing:Hashing input: {_Test123}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}",
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}"])
            # repr of an unitialized class is just the import path
            self.assertNotEqual(logs.output[2], f"DEBUG:simpleml.persistables.hashing:Hashing input: {repr(_Test123())}")

    def test_pandas_series_hashing(self):
        # series
        for d, expected_final_hash in zip(
            [range(20), ['a'], [1]],
            [7008921389990319782, -4496393130729816112, 6238072747940578789]
        ):
            with self.subTest(d=d, expected_final_hash=expected_final_hash):
                with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
                    # input/output
                    data = pd.Series(d)
                    with self.subTest():
                        self.assertEqual(CustomHasherMixin.custom_hasher(data), expected_final_hash)

                    # internal behavior
                    # hash series
                    self.assertEqual(
                        logs.output,
                        [f"DEBUG:simpleml.persistables.hashing:Hashing input: {data}",
                         "DEBUG:simpleml.persistables.hashing:hash type: <class 'pandas.core.series.Series'>",
                         f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}"])

    def test_pandas_frame_hashing(self):
        # frame
        for d, expected_final_hash in zip(
            [[range(10), range(10)], ['a'], [1]],
            [6716675364149054294, 5694802365760992243, -7087755961261762286]
        ):
            with self.subTest(d=d, expected_final_hash=expected_final_hash):
                with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
                    # input/output
                    data = pd.DataFrame(d)
                    with self.subTest():
                        self.assertEqual(CustomHasherMixin.custom_hasher(data), expected_final_hash)

                    # internal behavior
                    # hash dataframe
                    self.assertEqual(
                        logs.output,
                        [f"DEBUG:simpleml.persistables.hashing:Hashing input: {data}",
                         "DEBUG:simpleml.persistables.hashing:hash type: <class 'pandas.core.frame.DataFrame'>",
                         f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}"])

    def test_none_hashing(self):
        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            data = None
            expected_final_hash = -12345678987654321
            with self.subTest():
                self.assertEqual(CustomHasherMixin.custom_hasher(data), expected_final_hash)

            # internal behavior
            # hash None
            self.assertEqual(
                logs.output,
                [f"DEBUG:simpleml.persistables.hashing:Hashing input: {data}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'NoneType'>",
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}"])

    def test_complex_list_hashing(self):
        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            data = [
                'a',
                2,
                ['b', 3],
                {'d': 4},
                lambda: 0,
                pd.Series(['a']),
                pd.DataFrame([1])
            ]
            expected_final_hash = '68e95c072ffb1a8271e7e472f9fee504'
            with self.subTest():
                self.assertEqual(CustomHasherMixin.custom_hasher(data), expected_final_hash)

            # internal behavior
            # hash list -> hash items in list
            self.assertEqual(
                logs.output,
                [f"DEBUG:simpleml.persistables.hashing:Hashing input: {data}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'list'>",

                 # primitives
                 "DEBUG:simpleml.persistables.hashing:Hashing input: a",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 0357109b163771392cc674173d921e4b",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 2",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 76f34d73a1a6753d1243c9ba0afe3457",

                 # simple containers
                 "DEBUG:simpleml.persistables.hashing:Hashing input: ['b', 3]",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'list'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: b",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 10b474053f957b5c70dd5f01c695b8a0",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 3",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 56615ea01687173ebab08c915ad7e500",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 38b1de0299d81decb1341f9f2bfb4c8b",

                 "DEBUG:simpleml.persistables.hashing:Hashing input: {'d': 4}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'dict'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: ('d', 4)",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: d",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 5adbbd6cebbee97eda238235075de7ea",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 4",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: a8216e26a2093b48a0b7c57159313c8e",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 0bd9aca51ddaab2f96485637ec4c21ed",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 21065bb299df9d8a902754661f1dcf08",

                 # functions
                 f"DEBUG:simpleml.persistables.hashing:Hashing input: {data[4]}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'function'>",
                 # source inspection pulls the line the function is defined on with all whitespace
                 # depending on source, this could be more variables than just the function
                 "DEBUG:simpleml.persistables.hashing:Hashing input:                 lambda: 0,\n",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 1f55d5d00641bc583fef1c244a94116d",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 1f55d5d00641bc583fef1c244a94116d",

                 # data
                 f"DEBUG:simpleml.persistables.hashing:Hashing input: {data[5]}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'pandas.core.series.Series'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: -4496393130729816112",

                 f"DEBUG:simpleml.persistables.hashing:Hashing input: {data[6]}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'pandas.core.frame.DataFrame'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: -7087755961261762286",

                 # Final
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}"])

    def test_primitive_list_hashing(self):
        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            data = ['a', 2, ['b', 3], {'d': 4}]
            expected_final_hash = 'c3ee3ea76093a4ffa266010db2a19748'
            with self.subTest():
                self.assertEqual(CustomHasherMixin.custom_hasher(data), expected_final_hash)

            # internal behavior
            # hash list -> hash items in list
            self.assertEqual(
                logs.output,
                [f"DEBUG:simpleml.persistables.hashing:Hashing input: {data}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'list'>",

                 # primitives
                 "DEBUG:simpleml.persistables.hashing:Hashing input: a",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 0357109b163771392cc674173d921e4b",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 2",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 76f34d73a1a6753d1243c9ba0afe3457",

                 # simple containers
                 "DEBUG:simpleml.persistables.hashing:Hashing input: ['b', 3]",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'list'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: b",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 10b474053f957b5c70dd5f01c695b8a0",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 3",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 56615ea01687173ebab08c915ad7e500",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 38b1de0299d81decb1341f9f2bfb4c8b",

                 "DEBUG:simpleml.persistables.hashing:Hashing input: {'d': 4}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'dict'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: ('d', 4)",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: d",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 5adbbd6cebbee97eda238235075de7ea",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 4",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: a8216e26a2093b48a0b7c57159313c8e",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 0bd9aca51ddaab2f96485637ec4c21ed",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 21065bb299df9d8a902754661f1dcf08",

                 # Final
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}"])

    def test_pandas_list_hashing(self):
        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            data = [pd.Series(['a']), pd.DataFrame([1])]
            expected_final_hash = '9357fb780e7774f3426bc93d5eccdcc0'
            with self.subTest():
                self.assertEqual(CustomHasherMixin.custom_hasher(data), expected_final_hash)

            # internal behavior
            # hash list -> hash items in list
            self.assertEqual(
                logs.output,
                [f"DEBUG:simpleml.persistables.hashing:Hashing input: {data}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'list'>",

                 # data
                 f"DEBUG:simpleml.persistables.hashing:Hashing input: {data[0]}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'pandas.core.series.Series'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: -4496393130729816112",

                 f"DEBUG:simpleml.persistables.hashing:Hashing input: {data[1]}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'pandas.core.frame.DataFrame'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: -7087755961261762286",

                 # Final
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}"])

    def test_complex_dict_hashing(self):
        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            data = {
                'a': 2,
                'b': ['b', 3],
                'c': {'d': 4},
                'd': lambda: 0,
                'e': pd.Series(['a']),
                'f': pd.DataFrame([1])
            }

            expected_final_hash = '1cc5ab5d0c77f755358fe7f4d77ea04a'
            with self.subTest():
                self.assertEqual(CustomHasherMixin.custom_hasher(data), expected_final_hash)

            # internal behavior
            # hash dict -> hash items in dict
            self.assertEqual(
                logs.output,
                [f"DEBUG:simpleml.persistables.hashing:Hashing input: {data}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'dict'>",

                 # primitives
                 "DEBUG:simpleml.persistables.hashing:Hashing input: ('a', 2)",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: a",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 0357109b163771392cc674173d921e4b",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 2",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 76f34d73a1a6753d1243c9ba0afe3457",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 4168a931adf69a5c1cfd58cc89a5934b",

                 # simple containers
                 "DEBUG:simpleml.persistables.hashing:Hashing input: ('b', ['b', 3])",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: b",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 10b474053f957b5c70dd5f01c695b8a0",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: ['b', 3]",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'list'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: b",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 10b474053f957b5c70dd5f01c695b8a0",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 3",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 56615ea01687173ebab08c915ad7e500",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 38b1de0299d81decb1341f9f2bfb4c8b",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: ddfeb8c7d0f3b5e186ea6d5f75dc3a42",

                 "DEBUG:simpleml.persistables.hashing:Hashing input: ('c', {'d': 4})",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: c",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: eb5af44d447eeee22659894e100629ba",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: {'d': 4}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'dict'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: ('d', 4)",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: d",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 5adbbd6cebbee97eda238235075de7ea",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 4",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: a8216e26a2093b48a0b7c57159313c8e",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 0bd9aca51ddaab2f96485637ec4c21ed",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 21065bb299df9d8a902754661f1dcf08",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 23b65131a3c1e7692718ce5e16dbc6e1",

                 # functions
                 f"DEBUG:simpleml.persistables.hashing:Hashing input: ('d', {data['d']})",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: d",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 5adbbd6cebbee97eda238235075de7ea",
                 f"DEBUG:simpleml.persistables.hashing:Hashing input: {data['d']}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'function'>",
                 # source inspection pulls the line the function is defined on with all whitespace
                 # depending on source, this could be more variables than just the function
                 "DEBUG:simpleml.persistables.hashing:Hashing input:                 'd': lambda: 0,\n",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: bface6eb385c3eda922dae2ea0b1392d",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: bface6eb385c3eda922dae2ea0b1392d",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: db969ff10c6c237542b1244b2a54d4c3",

                 # data
                 f"DEBUG:simpleml.persistables.hashing:Hashing input: ('e', {data['e']})",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: e",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: f97a2d5131312082a54b26e764026dfd",
                 f"DEBUG:simpleml.persistables.hashing:Hashing input: {data['e']}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'pandas.core.series.Series'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: -4496393130729816112",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 022f7f3c9c3c4f477b8537dce4eb7b11",

                 f"DEBUG:simpleml.persistables.hashing:Hashing input: ('f', {data['f']})",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: f",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: d6a88b3c515fcfac7a70b4ee89ecc94d",
                 f"DEBUG:simpleml.persistables.hashing:Hashing input: {data['f']}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'pandas.core.frame.DataFrame'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: -7087755961261762286",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 214e5e5e60ff60baee6174e1846e0625",

                 # Final
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}"])

    def test_string_hashing(self):
        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            data = 'a'
            expected_final_hash = '0357109b163771392cc674173d921e4b'
            with self.subTest():
                self.assertEqual(CustomHasherMixin.custom_hasher(data), expected_final_hash)

            # internal behavior
            self.assertEqual(
                logs.output,
                [f"DEBUG:simpleml.persistables.hashing:Hashing input: {data}",
                 f"DEBUG:simpleml.persistables.hashing:hash type: {type(data)}",
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}"])

    def test_int_hashing(self):
        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            data = 2
            expected_final_hash = '76f34d73a1a6753d1243c9ba0afe3457'
            with self.subTest():
                self.assertEqual(CustomHasherMixin.custom_hasher(data), expected_final_hash)

            # internal behavior
            self.assertEqual(
                logs.output,
                [f"DEBUG:simpleml.persistables.hashing:Hashing input: {data}",
                 f"DEBUG:simpleml.persistables.hashing:hash type: {type(data)}",
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}"])

    def test_simple_list_hashing(self):
        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            data = ['b', 3]
            expected_final_hash = '38b1de0299d81decb1341f9f2bfb4c8b'
            with self.subTest():
                self.assertEqual(CustomHasherMixin.custom_hasher(data), expected_final_hash)

            # internal behavior
            self.assertEqual(
                logs.output,
                [f"DEBUG:simpleml.persistables.hashing:Hashing input: {data}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'list'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: b",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 10b474053f957b5c70dd5f01c695b8a0",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 3",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 56615ea01687173ebab08c915ad7e500",
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}"])

    def test_simple_dict_hashing(self):
        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            data = {'d': 4}
            expected_final_hash = '21065bb299df9d8a902754661f1dcf08'
            with self.subTest():
                self.assertEqual(CustomHasherMixin.custom_hasher(data), expected_final_hash)

            # internal behavior
            self.assertEqual(
                logs.output,
                [f"DEBUG:simpleml.persistables.hashing:Hashing input: {data}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'dict'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: ('d', 4)",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: d",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 5adbbd6cebbee97eda238235075de7ea",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 4",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: a8216e26a2093b48a0b7c57159313c8e",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 0bd9aca51ddaab2f96485637ec4c21ed",
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}"])

    def test_lambda_hashing(self):
        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            def data():
                return 0
            expected_final_hash = 'd7ab3b20053da4fb93531950ad4ffb66'
            with self.subTest():
                self.assertEqual(CustomHasherMixin.custom_hasher(data), expected_final_hash)

            # internal behavior
            self.assertEqual(
                logs.output,
                [f"DEBUG:simpleml.persistables.hashing:Hashing input: {data}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'function'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input:             def data():\n                return 0\n",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}",
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}"])

    def test_empty_pandas_series_hashing(self):
        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            data = pd.Series()
            expected_final_hash = 0
            with self.subTest():
                self.assertEqual(CustomHasherMixin.custom_hasher(data), expected_final_hash)

            # internal behavior
            self.assertEqual(
                logs.output,
                [f"DEBUG:simpleml.persistables.hashing:Hashing input: {data}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'pandas.core.series.Series'>",
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}"])

    def test_empty_pandas_dataframe_hashing(self):
        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            data = pd.DataFrame()
            expected_final_hash = 0
            with self.subTest():
                self.assertEqual(CustomHasherMixin.custom_hasher(data), expected_final_hash)

            # internal behavior
            self.assertEqual(
                logs.output,
                [f"DEBUG:simpleml.persistables.hashing:Hashing input: {data}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'pandas.core.frame.DataFrame'>",
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}"])


class DeterministicHasherTests(unittest.TestCase):
    def test_tuple_hash(self):
        '''
        set/tuple/list
        '''
        data = ('0357109b163771392cc674173d921e4b', '76f34d73a1a6753d1243c9ba0afe3457', '38b1de0299d81decb1341f9f2bfb4c8b', '21065bb299df9d8a902754661f1dcf08')
        expected_hash = 'c3ee3ea76093a4ffa266010db2a19748'
        self.assertEqual(deterministic_hash(data), expected_hash)


if __name__ == '__main__':
    unittest.main(verbosity=2)
