'''
Directory for Processed, or traditional datasets. In situations of clean,
representative data, this can be used directly for modeling purposes.
Otherwise, a `raw dataset` needs to be created with a `dataset pipeline`
to transform it into the processed form.


Import modules to register class names in global registry
'''
from . import base_processed_dataset


__author__ = 'Elisha Yadgaran'
