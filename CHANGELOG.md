## Change Log


### 0.8.1 (2020-05-11)
- Python 3.5 Dependencies
- Binary Classification Metric patches

### 0.8.0 (2020-03-15)
- EOL Python 2 support

### 0.7.2 (2019-12-01)
- SSH Tunnel Workflow

### 0.7.1 (2019-10-13)
- Sklearn Pipeline patch
- Changed default log level to INFO

### 0.7 (2019-10-07)
- Thread-safe Keras Sequence dataset splits
- Additional Seq2Seq models
- Bastion tunneling support for SSH db connections
- Explicit modules for constants and imports
- Additional base classes for database connections (plain and alembic)
- Database independent sqlachemy types
- Switched pickle library from dill to cloudpickle
- SQLite support
- Changed default DB connection to SQLite

### 0.6 (2019-06-19)
- Full database initialization with alembic
- DB schema validation on start
- Main configuration file for all credentials
- Drop official support for python 3.4
- Automatic handling of no data operations
- Remaining cloud provider support
- Feature metadata for classification models
- Runtime environment validation
- Add Split and SplitContainer objects
- Simplejson dependency
- Pipeline generator support
- Library specific model base classes
- Generalized database connection classes

### 0.5 (2019-02-17)
- Default identity pipeline
- Alembic integration for database migration
- Standardized model inheritance pattern
- Condensed pandas split dataframes into single df
- Remaining classification metrics
- Updated schema with hash datatype
- Updated hash to use joblib code, consistent across initializations
- Generator pipeline and fitted kwarg
- Dropped base prefixes
- Moved composed subclasses to inits
- Unified datasets and pipelines

### 0.4 (2019-01-04)
- Keras Seq2Seq support
- Keras model support
- Minimized required installation dependencies
- Abstract base classes
- Complex object JSON serialization
- Python 3 compatibility
- Travis and tox for CI/testing

### 0.3 (2018-12-01)
- Save mixins for external files
- Keras sequential model support
- hdf5 binary saving mechanism
- numpy hashing support
- Some tests

### 0.2 (2018-09-09)
- Error classes
- Sphinx documentation
- Classification metrics
- Persistable creators and loaders
- Scikit-Learn classifiers
- Dataset cross validation split mixins
- Hashing mixin

### 0.1
- Base persistable classes - raw_dataset, dataset_pipelines, datasets, pipelines, models, metrics
- Project structure
- Auto registering class imports
- Database utilities
