'''
Hashing related tests
'''

__author__ = 'Elisha Yadgaran'


import unittest

import pandas as pd
from simpleml._external.joblib import hash as deterministic_hash
from simpleml.persistables.hashing import CustomHasherMixin


class _Test123(object):
    random_attribute = 'abc'

    def __init__(self):
        pass

    def fancy_method(self):
        pass

    def __repr__(self):
        return 'pretty repr of test class'


class CustomHasherTests(unittest.TestCase):
    '''
    Hashing tests for consistency across environment and machines.
    Expectations generated on Mac running python 3.7

    Tests trace recursive behavior via log assertions
    '''

    def test_initialized_class_hashing(self):
        '''
        Hashes the initialized object as (name, __dict__)
        '''

        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            hash_object = _Test123()
            self.maxDiff = None

            # results are sensitive to entrypoint (relative path names)
            if __name__ == 'simpleml.tests.unit.test_hashing':
                # entry from loader
                # input/output
                expected_final_hash = '8e8c5f11154ccf2eda948ab98f468bf9'
                expected_logs = [
                    "DEBUG:simpleml.persistables.hashing:Hashing input: pretty repr of test class",
                    "DEBUG:simpleml.persistables.hashing:hash type: <class 'simpleml.tests.unit.test_hashing._Test123'>",
                    "DEBUG:simpleml.persistables.hashing:Hashing input: (<class 'simpleml.tests.unit.test_hashing._Test123'>, {})",
                    "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                    "DEBUG:simpleml.persistables.hashing:Hashing input: <class 'simpleml.tests.unit.test_hashing._Test123'>",
                    "DEBUG:simpleml.persistables.hashing:hash type: <class 'type'>",
                    "WARNING:simpleml.persistables.hashing:Hashing class import path for <class 'simpleml.tests.unit.test_hashing._Test123'>, if a fully qualified import path is not used, calling again from a different location will yield different results!",
                    "DEBUG:simpleml.persistables.hashing:Hashing input: simpleml.tests.unit.test_hashing._Test123",
                    "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                    "DEBUG:simpleml.persistables.hashing:Hashing output: efc89d254a441c047df389223d0f14fc, <class 'str'>",
                    "DEBUG:simpleml.persistables.hashing:Hashing output: efc89d254a441c047df389223d0f14fc, <class 'str'>",
                    'DEBUG:simpleml.persistables.hashing:Hashing input: {}',
                    "DEBUG:simpleml.persistables.hashing:hash type: <class 'dict'>",
                    "DEBUG:simpleml.persistables.hashing:Hashing output: d41d8cd98f00b204e9800998ecf8427e, <class 'str'>",
                    f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'str'>",
                    f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'str'>"
                ]

            elif __name__ == '__main__':
                # entry from this file
                # input/output
                expected_final_hash = '0399fcca26cf14ec1b3e31b69ca2397e'
                expected_logs = [
                    "DEBUG:simpleml.persistables.hashing:Hashing input: pretty repr of test class",
                    "DEBUG:simpleml.persistables.hashing:hash type: <class '__main__._Test123'>",
                    "DEBUG:simpleml.persistables.hashing:Hashing input: (<class '__main__._Test123'>, {})",
                    "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                    "DEBUG:simpleml.persistables.hashing:Hashing input: <class '__main__._Test123'>",
                    "DEBUG:simpleml.persistables.hashing:hash type: <class 'type'>",
                    "WARNING:simpleml.persistables.hashing:Hashing class import path for <class '__main__._Test123'>, if a fully qualified import path is not used, calling again from a different location will yield different results!",
                    "DEBUG:simpleml.persistables.hashing:Hashing input: __main__._Test123",
                    "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                    "DEBUG:simpleml.persistables.hashing:Hashing output: 1ec00bc22a3c72500ab551cbb2f9d520, <class 'str'>",
                    "DEBUG:simpleml.persistables.hashing:Hashing output: 1ec00bc22a3c72500ab551cbb2f9d520, <class 'str'>",
                    'DEBUG:simpleml.persistables.hashing:Hashing input: {}',
                    "DEBUG:simpleml.persistables.hashing:hash type: <class 'dict'>",
                    "DEBUG:simpleml.persistables.hashing:Hashing output: d41d8cd98f00b204e9800998ecf8427e, <class 'str'>",
                    f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'str'>",
                    f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'str'>"
                ]

            with self.subTest():
                self.assertEqual(CustomHasherMixin.custom_hasher(hash_object), expected_final_hash)

            self.assertEqual(logs.output, expected_logs)

    def test_uninitialized_class_hashing(self):
        '''
        Hashes the repr(cls) for initialized objects
        '''

        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            hash_object = _Test123
            self.maxDiff = None

            # results are sensitive to entrypoint (relative path names)
            if __name__ == 'simpleml.tests.unit.test_hashing':
                # entry from loader
                # input/output
                expected_final_hash = 'efc89d254a441c047df389223d0f14fc'
                expected_logs = [
                    "DEBUG:simpleml.persistables.hashing:Hashing input: <class 'simpleml.tests.unit.test_hashing._Test123'>",
                    "DEBUG:simpleml.persistables.hashing:hash type: <class 'type'>",
                    "WARNING:simpleml.persistables.hashing:Hashing class import path for <class 'simpleml.tests.unit.test_hashing._Test123'>, if a fully qualified import path is not used, calling again from a different location will yield different results!",
                    "DEBUG:simpleml.persistables.hashing:Hashing input: simpleml.tests.unit.test_hashing._Test123",
                    "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                    f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'str'>",
                    f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'str'>"
                ]

            elif __name__ == '__main__':
                # entry from this file
                # input/output
                expected_final_hash = '1ec00bc22a3c72500ab551cbb2f9d520'
                expected_logs = [
                    "DEBUG:simpleml.persistables.hashing:Hashing input: <class '__main__._Test123'>",
                    "DEBUG:simpleml.persistables.hashing:hash type: <class 'type'>",
                    "WARNING:simpleml.persistables.hashing:Hashing class import path for <class '__main__._Test123'>, if a fully qualified import path is not used, calling again from a different location will yield different results!",
                    "DEBUG:simpleml.persistables.hashing:Hashing input: __main__._Test123",
                    "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                    f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'str'>",
                    f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'str'>"
                ]

            with self.subTest():
                self.assertEqual(CustomHasherMixin.custom_hasher(hash_object), expected_final_hash)

            self.assertEqual(logs.output, expected_logs)

    def test_uninitialized_class_dict_hashing(self):
        '''
        Hashes just class attributes (input via cls.__dict__)
        Recursively includes all public methods and class attributes
        '''

        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            expected_final_hash = 'c7317170afd08252742af30eb98fe2d3'
            with self.subTest():
                self.assertEqual(CustomHasherMixin.custom_hasher(_Test123.__dict__), expected_final_hash)

            # internal behavior
            # hash class dict -> hash dict
            self.maxDiff = None
            self.assertEqual(
                logs.output,
                [f"DEBUG:simpleml.persistables.hashing:Hashing input: {_Test123.__dict__}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'dict'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: ('random_attribute', 'abc')",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                 'DEBUG:simpleml.persistables.hashing:Hashing input: random_attribute',
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 2a0611fe4463747f0ec29cd5ad5664ef, <class 'str'>",
                 'DEBUG:simpleml.persistables.hashing:Hashing input: abc',
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 900150983cd24fb0d6963f7d28e17f72, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: a5cccdfa42200d663c4f62f18fe22af7, <class 'str'>",
                 f"DEBUG:simpleml.persistables.hashing:Hashing input: ('fancy_method', {_Test123.fancy_method})",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                 'DEBUG:simpleml.persistables.hashing:Hashing input: fancy_method',
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 702b2a8795c39644af3dfc8ad728f918, <class 'str'>",
                 f'DEBUG:simpleml.persistables.hashing:Hashing input: {_Test123.fancy_method}',
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'function'>",
                 'DEBUG:simpleml.persistables.hashing:Hashing input:     def fancy_method(self):\n        pass\n',
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: ee31cc150ab1b82f7cd90ee978eb4970, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: ee31cc150ab1b82f7cd90ee978eb4970, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: a32e10936ab14ba8a9afa0527229fb7f, <class 'str'>",
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'str'>"])

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
                         f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'int'>"])

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
                         f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'int'>"])

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
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'int'>"])

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
            expected_final_hash = '2be1e4c1f34ee1614844b6b5130052d0'
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
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 0cc175b9c0f1b6a831c399e269772661, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 2",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 4753b753264f02d409ef7e2fa734d1e5, <class 'str'>",

                 # simple containers
                 "DEBUG:simpleml.persistables.hashing:Hashing input: ['b', 3]",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'list'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: b",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 92eb5ffee6ae2fec3ad71c777531578f, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 3",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: c01ef2d65e504ea354c5bf4f5b6f6329, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: bf00e62763be22f17074498f35a68302, <class 'str'>",

                 "DEBUG:simpleml.persistables.hashing:Hashing input: {'d': 4}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'dict'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: ('d', 4)",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: d",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 8277e0910d750195b448797616e091ad, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 4",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 038835f45126b13749d59afa4382ec30, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 2bbf775e58a10239cb79016c7ae0ec92, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 63c3302ff7ac527023a43dd85cbb92e1, <class 'str'>",

                 # functions
                 f"DEBUG:simpleml.persistables.hashing:Hashing input: {data[4]}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'function'>",
                 # source inspection pulls the line the function is defined on with all whitespace
                 # depending on source, this could be more variables than just the function
                 "DEBUG:simpleml.persistables.hashing:Hashing input:                 lambda: 0,\n",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 7c7ecb893a2bdd05739b7fc600fda7e4, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 7c7ecb893a2bdd05739b7fc600fda7e4, <class 'str'>",

                 # data
                 f"DEBUG:simpleml.persistables.hashing:Hashing input: {data[5]}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'pandas.core.series.Series'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: -4496393130729816112, <class 'int'>",

                 f"DEBUG:simpleml.persistables.hashing:Hashing input: {data[6]}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'pandas.core.frame.DataFrame'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: -7087755961261762286, <class 'int'>",

                 # Final
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'str'>"])

    def test_primitive_list_hashing(self):
        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            data = ['a', 2, ['b', 3], {'d': 4}]
            expected_final_hash = 'a0a531f7754274ea2fe57fefce20a55e'
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
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 0cc175b9c0f1b6a831c399e269772661, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 2",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 4753b753264f02d409ef7e2fa734d1e5, <class 'str'>",

                 # simple containers
                 "DEBUG:simpleml.persistables.hashing:Hashing input: ['b', 3]",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'list'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: b",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 92eb5ffee6ae2fec3ad71c777531578f, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 3",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: c01ef2d65e504ea354c5bf4f5b6f6329, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: bf00e62763be22f17074498f35a68302, <class 'str'>",

                 "DEBUG:simpleml.persistables.hashing:Hashing input: {'d': 4}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'dict'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: ('d', 4)",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: d",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 8277e0910d750195b448797616e091ad, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 4",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 038835f45126b13749d59afa4382ec30, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 2bbf775e58a10239cb79016c7ae0ec92, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 63c3302ff7ac527023a43dd85cbb92e1, <class 'str'>",

                 # Final
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'str'>"])

    def test_pandas_list_hashing(self):
        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            data = [pd.Series(['a']), pd.DataFrame([1])]
            expected_final_hash = '58d577105165dfc792672f4e430f2b0a'
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
                 "DEBUG:simpleml.persistables.hashing:Hashing output: -4496393130729816112, <class 'int'>",

                 f"DEBUG:simpleml.persistables.hashing:Hashing input: {data[1]}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'pandas.core.frame.DataFrame'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: -7087755961261762286, <class 'int'>",

                 # Final
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'str'>"])

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

            expected_final_hash = '86e34938e508c4e41143331423d35135'
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
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 0cc175b9c0f1b6a831c399e269772661, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 2",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 4753b753264f02d409ef7e2fa734d1e5, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: ac4e6d00fdf03922ad2785e91a749963, <class 'str'>",

                 # simple containers
                 "DEBUG:simpleml.persistables.hashing:Hashing input: ('b', ['b', 3])",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: b",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 92eb5ffee6ae2fec3ad71c777531578f, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: ['b', 3]",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'list'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: b",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 92eb5ffee6ae2fec3ad71c777531578f, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 3",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: c01ef2d65e504ea354c5bf4f5b6f6329, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: bf00e62763be22f17074498f35a68302, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 89204e715c58f21a1cb85ee468e932f6, <class 'str'>",

                 "DEBUG:simpleml.persistables.hashing:Hashing input: ('c', {'d': 4})",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: c",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 4a8a08f09d37b73795649038408b5f33, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: {'d': 4}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'dict'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: ('d', 4)",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: d",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 8277e0910d750195b448797616e091ad, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 4",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 038835f45126b13749d59afa4382ec30, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 2bbf775e58a10239cb79016c7ae0ec92, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 63c3302ff7ac527023a43dd85cbb92e1, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: bb623855eb965681e3be391b1d01df2d, <class 'str'>",

                 # functions
                 f"DEBUG:simpleml.persistables.hashing:Hashing input: ('d', {data['d']})",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: d",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 8277e0910d750195b448797616e091ad, <class 'str'>",
                 f"DEBUG:simpleml.persistables.hashing:Hashing input: {data['d']}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'function'>",
                 # source inspection pulls the line the function is defined on with all whitespace
                 # depending on source, this could be more variables than just the function
                 "DEBUG:simpleml.persistables.hashing:Hashing input:                 'd': lambda: 0,\n",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: c9a7e524abd9d6db4108a4314e7082d7, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: c9a7e524abd9d6db4108a4314e7082d7, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: ae338bc5442a052f122182363859534b, <class 'str'>",

                 # data
                 f"DEBUG:simpleml.persistables.hashing:Hashing input: ('e', {data['e']})",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: e",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: e1671797c52e15f763380b45e841ec32, <class 'str'>",
                 f"DEBUG:simpleml.persistables.hashing:Hashing input: {data['e']}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'pandas.core.series.Series'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: -4496393130729816112, <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 389d9f8d08ee8620fe0eeb9c55228418, <class 'str'>",

                 f"DEBUG:simpleml.persistables.hashing:Hashing input: ('f', {data['f']})",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'tuple'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: f",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 8fa14cdd754f91cc6554c9e71929cce7, <class 'str'>",
                 f"DEBUG:simpleml.persistables.hashing:Hashing input: {data['f']}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'pandas.core.frame.DataFrame'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: -7087755961261762286, <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: f91de4d68dee99f575029a1ee9a7a265, <class 'str'>",

                 # Final
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'str'>"])

    def test_string_hashing(self):
        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            data = 'a'
            expected_final_hash = '0cc175b9c0f1b6a831c399e269772661'
            with self.subTest():
                self.assertEqual(CustomHasherMixin.custom_hasher(data), expected_final_hash)

            # internal behavior
            self.assertEqual(
                logs.output,
                [f"DEBUG:simpleml.persistables.hashing:Hashing input: {data}",
                 f"DEBUG:simpleml.persistables.hashing:hash type: {type(data)}",
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'str'>"])

    def test_int_hashing(self):
        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            data = 2
            expected_final_hash = '4753b753264f02d409ef7e2fa734d1e5'
            with self.subTest():
                self.assertEqual(CustomHasherMixin.custom_hasher(data), expected_final_hash)

            # internal behavior
            self.assertEqual(
                logs.output,
                [f"DEBUG:simpleml.persistables.hashing:Hashing input: {data}",
                 f"DEBUG:simpleml.persistables.hashing:hash type: {type(data)}",
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'str'>"])

    def test_simple_list_hashing(self):
        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            data = ['b', 3]
            expected_final_hash = 'bf00e62763be22f17074498f35a68302'
            with self.subTest():
                self.assertEqual(CustomHasherMixin.custom_hasher(data), expected_final_hash)

            # internal behavior
            self.assertEqual(
                logs.output,
                [f"DEBUG:simpleml.persistables.hashing:Hashing input: {data}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'list'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: b",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 92eb5ffee6ae2fec3ad71c777531578f, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 3",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: c01ef2d65e504ea354c5bf4f5b6f6329, <class 'str'>",
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'str'>"])

    def test_simple_dict_hashing(self):
        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            data = {'d': 4}
            expected_final_hash = '63c3302ff7ac527023a43dd85cbb92e1'
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
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 8277e0910d750195b448797616e091ad, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input: 4",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'int'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 038835f45126b13749d59afa4382ec30, <class 'str'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing output: 2bbf775e58a10239cb79016c7ae0ec92, <class 'str'>",
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'str'>"])

    def test_lambda_hashing(self):
        with self.assertLogs(logger='simpleml.persistables.hashing', level='DEBUG') as logs:
            # input/output
            def data():
                return 0
            expected_final_hash = 'edf9b34707b6a63fd5ec95017e690f8f'
            with self.subTest():
                self.assertEqual(CustomHasherMixin.custom_hasher(data), expected_final_hash)

            # internal behavior
            self.assertEqual(
                logs.output,
                [f"DEBUG:simpleml.persistables.hashing:Hashing input: {data}",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'function'>",
                 "DEBUG:simpleml.persistables.hashing:Hashing input:             def data():\n                return 0\n",
                 "DEBUG:simpleml.persistables.hashing:hash type: <class 'str'>",
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'str'>",
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'str'>"])

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
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'int'>"])

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
                 f"DEBUG:simpleml.persistables.hashing:Hashing output: {expected_final_hash}, <class 'int'>"])


class MD5HasherTests(unittest.TestCase):
    def test_tuple_hash(self):
        '''
        set/tuple/list/dict/mappingproxy reduce to a tuple of hashes
        '''
        data = ('0357109b163771392cc674173d921e4b', '76f34d73a1a6753d1243c9ba0afe3457', '38b1de0299d81decb1341f9f2bfb4c8b', '21065bb299df9d8a902754661f1dcf08')
        expected_hash = '57ef70a19f5ecb8fc70d9f173d4f7740'
        self.assertEqual(CustomHasherMixin.md5_hasher(data), expected_hash)

    def test_string_hash(self):
        for data, expected_hash in {
            'abc': '900150983cd24fb0d6963f7d28e17f72',
            'ajfoh203949fja..!#@#@$@': '1cb83ecc9f13349b4b04eb4b4aa93d26',
            'AFAL;FADFKA;JAFIEFHIA': '33b899614e1e16b4a84646d435064f53',
            'extra_long_value_extra_long_value_extra_long_value_extra_long_value_extra_long_value_extra_long_value': 'e192f15053eb1e6fce7d0a0769280dc5'
        }.items():
            self.assertEqual(CustomHasherMixin.md5_hasher(data), expected_hash)

    def test_int_hash(self):
        for data, expected_hash in {
            12: '9788cdcdd2f907b2ba4c106e05db77dd',
            12756387463875648597426574256294765284528457465783: '4fceac488520aff1b49e0419ff29aef4',
            -76348735275375648597426574256294765284528457465783: 'a30cb8a201eec96790acd4057d8b5de0',
            122390254752894527495027528529045974590245949201071407541071504574910: 'bde13a1a7d7af2e3d3af1650e3af5d2e',
            -2898275276148014718015745618934189374190558956134871589634510914871105: '8aae8e41e85aa776b435dddefd741b6a'
        }.items():
            self.assertEqual(CustomHasherMixin.md5_hasher(data), expected_hash)

    def test_float_hash(self):
        for data, expected_hash in {
            0.045: '4ec2e10062562fe8ba5183cfee61dc7f',
            0.0981209867893243456787453211253689098765265778484245: '491007207b7b93336843ee3de9d64ec5'
        }.items():
            self.assertEqual(CustomHasherMixin.md5_hasher(data), expected_hash)

    def test_other_dtype_error(self):
        for dtype in ({}, [], set(), pd.DataFrame, None):
            with self.assertRaises(ValueError):
                CustomHasherMixin.md5_hasher(dtype)


class PickleHasherTests(unittest.TestCase):
    def test_tuple_hash(self):
        '''
        set/tuple/list/dict/mappingproxy reduce to a tuple of hashes
        '''
        data = ('0357109b163771392cc674173d921e4b', '76f34d73a1a6753d1243c9ba0afe3457', '38b1de0299d81decb1341f9f2bfb4c8b', '21065bb299df9d8a902754661f1dcf08')
        expected_hash = 'c3ee3ea76093a4ffa266010db2a19748'
        self.assertEqual(deterministic_hash(data), expected_hash)

    def test_string_hash(self):
        data = 'abc'
        expected_hash = 'a5a2f6c8adba6852e4d3888ce0c26016'
        self.assertEqual(deterministic_hash(data), expected_hash)

    def test_int_hash(self):
        data = 12
        expected_hash = 'feb1c5cac6acf399a62e281ca8aaac96'
        self.assertEqual(deterministic_hash(data), expected_hash)

    def test_float_hash(self):
        data = 0.045
        expected_hash = '900c461ea0f92e9dba4eaef616dbfd35'
        self.assertEqual(deterministic_hash(data), expected_hash)


if __name__ == '__main__':
    unittest.main(verbosity=2)
