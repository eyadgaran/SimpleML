Datasets
========

Uniqueness
----------
When creating a dataset persistable, there are a few limitations to ensure uniqueness.
The main one is that the data (or reference to it, depending on hashing rules) is
unambiguously captured in the hash. This allows different datasets to be compared
and even the same dataset to be referenced in different runtimes. A cascading effect
of this is that the data has to be accessible and unchanging over time - otherwise
the hash would be considered unstable and yield different experiment results during
different executions.

More concretely this means that a dataset cannot be defined uniquely by a sql
query or file based data generator. It has to also include the data, in a stable
form to ensure the same outcomes downstream. This is the main reason that the
default hashing routine includes the resulting data (at the cost of delayed hash
evaluation).

It is, however, possible to introduce a different form of dataset
iteration after the dataset has been "built" and saved. This has the advantage of
maintaining a reproducible data source while allowing custom iterations over that
data for later use. The API lists the commands and parameters to tune usage.

For situations where the reference to the data is enough of a guarantee (static
sql table, csv file, etc), the dataset class can be subclassed and the hash
overwritten.::

  class ReferenceHashedDataset(Dataset):
      def _hash(self):
          '''
          Change the hash to only include the data reference - or to whatever
          is desired

          Hash is the combination of the:
              1) Config
              2) Pipeline
          '''
          config = self.config
          if self.pipeline is not None:
              pipeline_hash = self.pipeline.hash_ or self.pipeline._hash()
          else:
              pipeline_hash = None

          return self.custom_hasher((config, pipeline_hash))


Labels
------
For Supervised tasks, labels need to be specified in the dataset. A design choice
was made here to consider the same datasets as different if they specified
different target labels. The motivation for that decision is based on the downstream
use of those datasets and the potential for confusion. If the resulting models
shared the same reference, there would be potential collisions and silent errors.

Labels are set via the `label_columns` parameter in the initializer and must be
passed as a list.


Constructing the "Dataframe"
----------------------------
The term "Dataframe" is used to reference the underlying data inside the dataset.
By no means does a pandas dataframe need to be used, but they are pretty useful...
See the dataset mixins for examples of different data structures.

Internally `self._external_file` is the attribute that needs to be set with the
data. The build_dataframe method is called to marshal the data and set `self._external_file`.
Override that method to implement the desired loading mechanism.


Mixins
------
Different Mixin classes are available to interface with the underlying data
structures. They follow a standard pattern so consumers (pipelines) can remain
blissfully unaware of exactly how the data is being surfaced. This is also where
future extension to currently unsupported dataframes can happen.

The basic form of a dataset mixin is::

  class AbstractDatasetMixin(object):
      @property
      def X(self):
          '''
          Return the subset that isn't in the target labels
          '''
          raise NotImplementedError

      @property
      def y(self):
          '''
          Return the target label columns
          '''
          raise NotImplementedError

      def get(self, column, split):
          '''
          Unimplemented method to explicitly split X and y
          Must be implemented by subclasses
          '''
          raise NotImplementedError

      def get_feature_names(self):
          '''
          Should return a list of the features in the dataset
          '''
          raise NotImplementedError

Methods: `X`, `y`, `get`; all target retrieving some subset of the data. Implementations
exist for both pandas-like and numpy dataframes as well as composed dataset classes.

Method `get_feature_names` also returns a subset of the data, but most importantly
it only returns feature information for later analysis.


Extensibility
-------------
All SimpleML persistables are designed with extensibility in mind. The frameworks
and libraries of today are not necessarily going to continue to be the defacto.
Datasets are no different and contain hooks to emulate the same behavior with
different implementations.

Use the examples of Pandas and Numpy datasets for guidance on implementing a
new type of dataset.
