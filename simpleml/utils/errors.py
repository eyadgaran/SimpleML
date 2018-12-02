'''
Error classes
'''

__author__ = 'Elisha Yadgaran'


class SimpleMLError(Exception):
    def __str__(self):
        return self.message

class DatasetError(SimpleMLError):
    def __init__(self, *args, **kwargs):
        super(DatasetError, self).__init__(*args, **kwargs)
        custom_prefix = 'SimpleML Dataset Error: '
        self.message = custom_prefix + self.message


class PipelineError(SimpleMLError):
    def __init__(self, *args, **kwargs):
        super(PipelineError, self).__init__(*args, **kwargs)
        custom_prefix = 'SimpleML Pipeline Error: '
        self.message = custom_prefix + self.message


class ModelError(SimpleMLError):
    def __init__(self, *args, **kwargs):
        super(ModelError, self).__init__(*args, **kwargs)
        custom_prefix = 'SimpleML Model Error: '
        self.message = custom_prefix + self.message


class MetricError(SimpleMLError):
    def __init__(self, *args, **kwargs):
        super(MetricError, self).__init__(*args, **kwargs)
        custom_prefix = 'SimpleML Metric Error: '
        self.message = custom_prefix + self.message


class TrainingError(SimpleMLError):
    def __init__(self, *args, **kwargs):
        super(TrainingError, self).__init__(*args, **kwargs)
        custom_prefix = 'SimpleML Training Error: '
        self.message = custom_prefix + self.message


class ScoringError(SimpleMLError):
    def __init__(self, *args, **kwargs):
        super(ScoringError, self).__init__(*args, **kwargs)
        custom_prefix = 'SimpleML Scoring Error: '
        self.message = custom_prefix + self.message
