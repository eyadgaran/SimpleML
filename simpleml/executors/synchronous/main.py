class MainProcessExecutor(object):
    @staticmethod
    def process(*args, op, **kwargs):
        return op(*args, **kwargs)
