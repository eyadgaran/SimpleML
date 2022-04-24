'''
Helper classes to iterate splits
'''


__author__ = 'Elisha Yadgaran'


from typing import Any, Tuple, Union

import numpy as np
import pandas as pd

from simpleml.datasets.dataset_splits import Split
from simpleml.imports import Sequence
from simpleml.pipelines import Pipeline


def split_to_ordered_tuple(split: Split) -> Tuple:
    '''
    Helper to convert a split object into an ordered tuple of
    X, y, other
    '''
    return_objects = []
    X = split.X
    y = split.y

    if X is not None:
        return_objects.append(X)
    if y is not None:
        return_objects.append(y)

    for k, v in split.items():
        if k not in ('X', 'y') and v is not None:
            return_objects.append(v)
    return return_objects


class DataIterator(object):
    def __iter__(self):
        return self


'''
Python native implementation
'''


class PythonIterator(DataIterator):
    '''
    Pure python iterator. Converts a split object into a generator with defined
    batch sizes
    '''

    def __init__(self,
                 split: Split,
                 infinite_loop: bool = False,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 return_tuple: bool = False,
                 **kwargs):
        self.split = split
        self.infinite_loop = infinite_loop
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_tuple = return_tuple
        self.generate_indices()

    def generate_indices(self):
        # Extract indices to subsample from
        X = self.split.X
        if isinstance(X, pd.DataFrame):
            indices = X.index.tolist()
        elif isinstance(X, np.ndarray):
            indices = np.arange(X.shape[0])
        else:
            raise NotImplementedError

        self.dataset_size = X.shape[0]
        self.indices = indices

    def __iter__(self):
        # Loop through and sample indefinitely
        self.first_run = True
        self.current_index = 0

        return self

    def __next__(self) -> Union[Split, Tuple]:
        '''
        Turn a dataset split into a generator
        '''
        X = self.split.X
        y = self.split.y

        if self.dataset_size == 0:  # Return None
            raise StopIteration

        # Loop so that infinite batches can be generated
        if self.current_index >= self.dataset_size:
            if self.infinite_loop:
                self.current_index = 0
                self.first_run = False
            else:
                raise StopIteration

        # shuffle on new loops
        if self.current_index == 0 and self.shuffle and not self.first_run:
            self.indices = np.random.shuffle(self.indices)

        # next batch indices
        batch = self.indices[self.current_index:min(self.current_index + self.batch_size, self.dataset_size)]
        self.current_index += self.batch_size

        if y is not None and (isinstance(y, (pd.DataFrame, pd.Series)) and not y.empty):  # Supervised
            if isinstance(X, (pd.DataFrame, pd.Series)):
                split = Split(X=X.loc[batch], y=np.stack(y.loc[batch].squeeze().values))
            else:
                split = Split(X=X[batch], y=y[batch])
        else:  # Unsupervised
            if isinstance(X, (pd.DataFrame, pd.Series)):
                split = Split(X=X.loc[batch])
            else:
                split = Split(X=X[batch])

        if self.return_tuple:
            return split_to_ordered_tuple(split)
        else:
            return split


class PipelineTransformIterator(DataIterator):
    '''
    Wrapper utility to convert a pipeline transform operation into an iterator
    Transforms batch on iteration with provided pipeline
    '''

    def __init__(self,
                 pipeline: Pipeline,
                 data_iterator: DataIterator):
        self.pipeline = pipeline
        self.data_iterator = data_iterator

    def __iter__(self):
        # restart data iterator
        self.data_iterator = iter(self.data_iterator)
        return self

    def __next__(self) -> Union[Split, Tuple]:
        '''
        NOTE: Some downstream objects expect to consume a generator with a tuple of
        X, y, other... not a Split object, so an ordered tuple will be returned
        if the dataset iterator returns a tuple
        '''
        batch = next(self.data_iterator)
        if isinstance(batch, tuple):
            X = batch[0]
            return_tuple = True
        else:
            X = batch.X
            return_tuple = False

        output = self.pipeline.transform(X)

        if return_tuple:
            # Return input with X replaced by output (transformed X)
            # Contains y or other named inputs to propagate downstream
            # Explicitly order for *args input -- X, y, other...
            return tuple((X, *batch[1:]))

        else:
            return Split(X=output, **{k: v for k, v in batch.items() if k != 'X'})


'''
Keras Style implementation
'''


class DatasetSequence(Sequence):
    '''
    Sequence wrapper for internal datasets. Only used for raw data mapping so
    return type is internal `Split` object. Transformed sequences are used to
    conform with external input types (keras tuples)
    '''

    def __init__(self,
                 split: Split,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 return_tuple: bool = True,
                 **kwargs):
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_tuple = return_tuple
        self.generate_indices()

    def generate_indices(self) -> None:
        # Extract indices to subsample from
        X = self.split.X
        if isinstance(X, pd.DataFrame):
            indices = X.index.tolist()
        elif isinstance(X, np.ndarray):
            indices = np.arange(X.shape[0])
        else:
            raise NotImplementedError

        self.dataset_size: int = self.X.shape[0]
        if self.dataset_size == 0:  # Return None
            raise ValueError('Attempting to create sequence with no data')

        self.indices = indices

    @staticmethod
    def validated_split(split: Any) -> Any:
        '''
        Confirms data is valid, otherwise returns None (makes downstream checking
        simpler)
        '''
        if split is None:
            return None
        elif isinstance(split, (pd.DataFrame, pd.Series)) and split.empty:
            return None
        return split

    def __getitem__(self, index) -> Split:
        """Gets batch at position `index`.
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch
        """
        current_index = index * self.batch_size  # list index of batch start
        batch = self.indices[current_index:min(current_index + self.batch_size, self.dataset_size)]

        X = self.validated_split(self.split.X)
        y = self.validated_split(self.split.y)

        if y is not None:  # Supervised
            if isinstance(X, (pd.DataFrame, pd.Series)):
                split = Split(X=X.loc[batch], y=np.stack(y.loc[batch].squeeze().values))
            else:
                split = Split(X=X[batch], y=y[batch])
        else:  # Unsupervised
            if isinstance(X, (pd.DataFrame, pd.Series)):
                split = Split(X=X.loc[batch])
            else:
                split = Split(X=X[batch])

        if self.return_tuple:
            return split_to_ordered_tuple(split)
        else:
            return split

    def __len__(self):
        """Number of batch in the Sequence.
        # Returns
            The number of batches in the Sequence.
        """
        return int(np.ceil(len(self.dataset_size) / float(self.batch_size)))

    def on_epoch_end(self) -> None:
        """Method called at the end of every epoch.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)


class PipelineTransformSequence(Sequence):
    '''
    Nested sequence class to apply transforms on batches in real-time and forward
    through as the next batch
    '''

    def __init__(self,
                 pipeline: Pipeline,
                 dataset_sequence: DatasetSequence):
        self.pipeline = pipeline
        self.dataset_sequence = dataset_sequence

    def __getitem__(self, *args, **kwargs) -> Union[Split, Tuple]:
        '''
        Pass-through to dataset sequence - applies transform on data and returns transformed batch
        '''
        batch: Union[Tuple, Split] = self.dataset_sequence(*args, **kwargs)

        if isinstance(batch, tuple):
            X = batch[0]
            return_tuple = True
        else:
            X = batch.X
            return_tuple = False

        output = self.pipeline.transform(X)

        if return_tuple:
            # Return input with X replaced by output (transformed X)
            # Contains y or other named inputs to propagate downstream
            # Explicitly order for *args input -- X, y, other...
            return tuple((X, *batch[1:]))

        else:
            return Split(X=output, **{k: v for k, v in batch.items() if k != 'X'})

    def __len__(self):
        '''
        Pass-through. Returns number of batches in dataset sequence
        '''
        return len(self.dataset_sequence)

    def on_epoch_end(self) -> None:
        self.dataset_sequence.on_epoch_end()
