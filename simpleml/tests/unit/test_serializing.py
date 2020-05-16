'''
Serializing related tests
'''

__author__ = 'Elisha Yadgaran'

from simpleml.persistables.serializing import custom_dumps, custom_loads
import unittest


class TestClass(object):
    '''Something not natively JSON serializable'''
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __eq__(self, other):
        return all((
            self.a == other.a,
            self.b == other.b,
            self.c == other.c,
            self.d == other.d
        ))

    def __repr__(self):
        return "{}-{}-{}-{}".format(self.a, self.b, self.c, self.d)


class SerializingTests(unittest.TestCase):
    def test_top_level_serializing(self):
        obj = TestClass(1, 'a', 2, 'b')
        dumped = custom_dumps(obj)
        loaded = custom_loads(dumped)
        self.assertEqual(obj, loaded)

    def test_number_serializing(self):
        obj = 123
        dumped = custom_dumps(obj)
        loaded = custom_loads(dumped)
        self.assertEqual(obj, loaded)

    def test_list_serializing(self):
        obj = [123, 'abc', TestClass(1, 'a', 2, 'b')]
        dumped = custom_dumps(obj)
        loaded = custom_loads(dumped)
        self.assertEqual(obj, loaded)

    def test_nested_list_serializing(self):
        obj = [123, 'abc', TestClass(1, 'a', 2, 'b'),
               [234, 'def', TestClass(2, 'b', 3, 'c')]]
        dumped = custom_dumps(obj)
        loaded = custom_loads(dumped)
        self.assertEqual(obj, loaded)

    def test_dict_serializing_int_keys(self):
        # Unfortunately JSON doesnt support dict keys as ints so they will automatically
        # get converted. Hopefully this wont be an issue in SimpleML, but be aware...
        obj = {
            u'abc': TestClass(1, 'a', 2, 'b'),
            123: u'def',
            u'hij': [TestClass(3, 'c', 4, 'd'), u'klm', 456]
        }

        expected_obj = {
            u'abc': TestClass(1, 'a', 2, 'b'),
            '123': u'def',
            u'hij': [TestClass(3, 'c', 4, 'd'), u'klm', 456]
        }
        dumped = custom_dumps(obj)
        loaded = custom_loads(dumped)
        self.assertNotEqual(obj, loaded)
        self.assertEqual(loaded, expected_obj)

    def test_dict_serializing_no_int_keys(self):
        obj = {
            u'abc': TestClass(1, 'a', 2, 'b'),
            '123': u'def',
            u'hij': [TestClass(3, 'c', 4, 'd'), u'klm', 456]
        }

        dumped = custom_dumps(obj)
        loaded = custom_loads(dumped)
        self.assertEqual(obj, loaded)


if __name__ == '__main__':
    unittest.main(verbosity=2)
