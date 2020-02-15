Persistables
============

Persistables are the fundamental blocks of SimpleML objects. They are defined
as the abstract base that details the expected methods and attributes of all
inheriting subclasses.


SQLAlchemy
----------
All persistables are backed by database records. This provides the seamless
transition between development and production. The other major benefit is
a full record of all iterations allowing for powerful querying and improvement
tracking. The database orchestration is conducted via sqlalchemy for compatibility
with all supported databases.


Registry
--------
The gatekeeper syncing the database backend to the defined classes is the
SimpleML-Registry. As the name suggests, it gets injected automatically into all
classes that inherit from the base persistable and registers them on import.
This means that references can be stored in the database (not the actual classes)
and training can be scheduled using deferred execution and graph construction.
The only limitation here is that any referenced class needs to be imported into
the global namespace (and the registry) before it is invoked.

Standard practice to ensure all classes are imported is to source them in their
respective package __init__ files and import the top level module before training
or scoring.


Hashing
-------
SimpleML exposes an optional top-level attribute for each persistable called the
hash. The intent is to unambiguously capture the instance in order to compare
references. The downstream use cases extend from conditional training (only train
if new) to graph analytics (which models share the same datasets or pipelines).

A custom hashing method is included in the base persistable that is configured to
handle all data types consistently across installations. Each inheriting class
can then define which attributes to include in the hash. For use with conditional
training, it is essential that the hash only consist of properties that exist
upon initialization. If the hash is dependent on values that get modified during
the training process, it will be impossible to compare new instances without first
training them and comparing the resulting hashes.


Saving
------


Serializing
-----------
