'''
Metaclass tests - verifies expected construction and initialization
'''


import logging
import unittest

LOGGER = logging.getLogger('meta_test')


class Meta1(type):
    def __new__(cls, *args, **kwargs):
        LOGGER.info('enter Meta1 __new__')
        r = super(Meta1, cls).__new__(cls, *args, **kwargs)
        LOGGER.info('exit Meta1 __new__')
        return r

    def __init__(self, *args, **kwargs):
        LOGGER.info('enter Meta1 __init__')
        super().__init__(*args, **kwargs)
        LOGGER.info('exit Meta1 __init__')

    def __call__(self, *args, **kwargs):
        LOGGER.info('enter Meta1 __call__')
        r = super().__call__(*args, **kwargs)
        if hasattr(r, 'post_init'):
            LOGGER.info('found post')
            r.post_init()
        LOGGER.info('exit Meta1 __call__')
        return r


class Concrete1(object, metaclass=Meta1):
    def __new__(cls, *args, **kwargs):
        LOGGER.info('enter Concrete1 __new__')
        r = super(Concrete1, cls).__new__(cls, *args, **kwargs)
        LOGGER.info('exit Concrete1 __new__')
        return r

    def __init__(self, *args, **kwargs):
        LOGGER.info('enter Concrete1 __init__')
        super().__init__(*args, **kwargs)
        LOGGER.info('exit Concrete1 __init__')

    def __call__(*args, **kwargs):
        LOGGER.info('enter Concrete1 __call__')
        super().__call__(*args, **kwargs)
        LOGGER.info('exit Concrete1 __call__')

    def post_init(self):
        LOGGER.info('post init')


class Concrete2(object, metaclass=Meta1):
    def __new__(cls, *args, **kwargs):
        LOGGER.info('enter Concrete2 __new__')
        r = super(Concrete2, cls).__new__(cls, *args, **kwargs)
        LOGGER.info('exit Concrete2 __new__')
        return r

    def __init__(self, *args, **kwargs):
        LOGGER.info('enter Concrete2 __init__')
        super().__init__(*args, **kwargs)
        LOGGER.info('exit Concrete2 __init__')

    def __call__(*args, **kwargs):
        LOGGER.info('enter Concrete2 __call__')
        super().__call__(*args, **kwargs)
        LOGGER.info('exit Concrete2 __call__')


class Concrete3(object, metaclass=Meta1):
    def __init__(self, *args, **kwargs):
        LOGGER.info('enter Concrete3 __init__')
        super().__init__(*args, **kwargs)
        LOGGER.info('exit Concrete3 __init__')

    def __call__(*args, **kwargs):
        LOGGER.info('enter Concrete3 __call__')
        super().__call__(*args, **kwargs)
        LOGGER.info('exit Concrete3 __call__')


class MetaclassInheritanceTest(unittest.TestCase):
    '''
    Generic test structure. Focuses on inheritance pattern for
    imports and construction
    '''

    def test_meta1(self):
        '''
        Normal metaclass construction and initialization
        should raise an error
        '''
        with self.assertRaises(TypeError) as e:
            Meta1()
        self.assertEqual(str(e.exception), 'type.__new__() takes exactly 3 arguments (0 given)')

    def test_concrete1(self):
        '''
        Metaclass construction. Normal initialization. Post init call
        '''
        with self.assertLogs(logger='meta_test', level='INFO') as logs:
            with self.subTest():
                m = Concrete1()
                self.assertTrue(isinstance(m, Concrete1))
            expected_logs = [
                'INFO:meta_test:enter Meta1 __call__',
                'INFO:meta_test:enter Concrete1 __new__',
                'INFO:meta_test:exit Concrete1 __new__',
                'INFO:meta_test:enter Concrete1 __init__',
                'INFO:meta_test:exit Concrete1 __init__',
                'INFO:meta_test:found post',
                'INFO:meta_test:post init',
                'INFO:meta_test:exit Meta1 __call__',
            ]
            self.assertEqual(expected_logs, logs.output)

    def test_concrete2(self):
        '''
        Normal construction. Normal initialization. no post init
        '''
        with self.assertLogs(logger='meta_test', level='INFO') as logs:
            with self.subTest():
                m = Concrete2()
                self.assertTrue(isinstance(m, Concrete2))
            expected_logs = [
                'INFO:meta_test:enter Meta1 __call__',
                'INFO:meta_test:enter Concrete2 __new__',
                'INFO:meta_test:exit Concrete2 __new__',
                'INFO:meta_test:enter Concrete2 __init__',
                'INFO:meta_test:exit Concrete2 __init__',
                'INFO:meta_test:exit Meta1 __call__',
            ]
            self.assertEqual(expected_logs, logs.output)

    def test_concrete3(self):
        '''
        Metaclass construction. Normal initialization. no post init
        '''
        with self.assertLogs(logger='meta_test', level='INFO') as logs:
            with self.subTest():
                m = Concrete3()
                self.assertTrue(isinstance(m, Concrete3))
            expected_logs = [
                'INFO:meta_test:enter Meta1 __call__',
                'INFO:meta_test:enter Concrete3 __init__',
                'INFO:meta_test:exit Concrete3 __init__',
                'INFO:meta_test:exit Meta1 __call__',
            ]
            self.assertEqual(expected_logs, logs.output)

    def test_reconstruction_without_init_with_new_method(self):
        with self.assertLogs(logger='meta_test', level='INFO') as logs:
            with self.subTest():
                m = Concrete2.__new__(Concrete2)
                self.assertTrue(isinstance(m, Concrete2))
            expected_logs = [
                'INFO:meta_test:enter Concrete2 __new__',
                'INFO:meta_test:exit Concrete2 __new__',
            ]
            self.assertEqual(expected_logs, logs.output)

    def test_reconstruction_without_init_without_new_method(self):
        with self.assertRaises(AssertionError) as e, \
                self.assertLogs(logger='meta_test', level='INFO') as logs:
            with self.subTest():
                m = Concrete3.__new__(Concrete3)
                self.assertTrue(isinstance(m, Concrete3))
            expected_logs = []
            self.assertEqual(expected_logs, logs.output)
        self.assertEqual('no logs of level INFO or higher triggered on meta_test', str(e.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
