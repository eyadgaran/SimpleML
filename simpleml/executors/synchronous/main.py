"""
The default main process executor. Does not attempt any modification of
the function, passing straight through as called.
"""

__author__ = "Elisha Yadgaran"


from simpleml.executors.processor import AbstractBaseExecutor


class MainProcessExecutor(AbstractBaseExecutor):
    @staticmethod
    def process(*args, op, **kwargs):
        return op(*args, **kwargs)
