'''
Error classes
'''

__author__ = 'Elisha Yadgaran'


class SimpleMLError(Exception):
    def __str__(self):
        if hasattr(self, 'message'):
            return self.message
        return self.args[0]

class DatasetError(SimpleMLError):
    def __init__(self, *args, **kwargs):
        super(DatasetError, self).__init__(*args, **kwargs)
        custom_prefix = 'SimpleML Dataset Error: '
        self.message = custom_prefix + self.args[0]


class PipelineError(SimpleMLError):
    def __init__(self, *args, **kwargs):
        super(PipelineError, self).__init__(*args, **kwargs)
        custom_prefix = 'SimpleML Pipeline Error: '
        self.message = custom_prefix + self.args[0]


class ModelError(SimpleMLError):
    def __init__(self, *args, **kwargs):
        super(ModelError, self).__init__(*args, **kwargs)
        custom_prefix = 'SimpleML Model Error: '
        self.message = custom_prefix + self.args[0]


class MetricError(SimpleMLError):
    def __init__(self, *args, **kwargs):
        super(MetricError, self).__init__(*args, **kwargs)
        custom_prefix = 'SimpleML Metric Error: '
        self.message = custom_prefix + self.args[0]


class TrainingError(SimpleMLError):
    def __init__(self, *args, **kwargs):
        super(TrainingError, self).__init__(*args, **kwargs)
        custom_prefix = 'SimpleML Training Error: '
        self.message = custom_prefix + self.args[0]


class ScoringError(SimpleMLError):
    def __init__(self, *args, **kwargs):
        super(ScoringError, self).__init__(*args, **kwargs)
        custom_prefix = 'SimpleML Scoring Error: '
        self.message = custom_prefix + self.args[0]
