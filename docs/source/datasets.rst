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
