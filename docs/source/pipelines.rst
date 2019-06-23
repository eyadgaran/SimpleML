Pipelines
=========


Dataset Reference
-----------------
When creating a pipeline persistable, the referenced dataset is assumed to exist
already. This subtle but important detail means that generators or other lazy
consumption mechanisms cannot be used as part of the initial creation. When transforming
after for model use, it is possible to introduce a different form of dataset
iteration. The API lists the commands and parameters to tune usage.
