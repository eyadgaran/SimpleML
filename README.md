[![PyPI version](https://badge.fury.io/py/simpleml.svg)](https://badge.fury.io/py/simpleml)
![PyPI pyversions](https://img.shields.io/pypi/pyversions/simpleml.svg)
![Docs](https://readthedocs.org/projects/simpleml/badge/?version=stable)
![Build Status](https://travis-ci.org/eyadgaran/SimpleML.svg?branch=master)
![Coverage Status](https://coveralls.io/repos/github/eyadgaran/SimpleML/badge.svg?branch=master)
![HitCount](http://hits.dwyl.io/eyadgaran/simpleml.svg)

# SimpleML
Machine learning that just works, for effortless production applications

Documentation: [simpleml.readthedocs.io](https://simpleml.readthedocs.io)

Installation:  `pip install simpleml`

# History
SimpleML started as a persistence solution to simplify some of the most common pain points in new modeling projects. It
offered an abstraction layer to implicitly version, persist, and load training iterations to make productionalizing
a project an effortless process. Extensibility is central to the design and is therefore compatible out of the box with
modeling libraries (Scikit-Learn, Tensorflow/Keras, etc) and algorithms, making it a low overhead drop-in
complement to workflows.

As the ML ops space has grown, more solutions are being offered to manage particular pain points or conform to
opinionated development and release patterns. These patterns while immensely powerful are rigid and not always the
ideal implementation for projects (and definitely not amenable to blending multiple frameworks in a project).
SimpleML is also growing to address this gap by evolving from a persistence framework to an ML management framework.
The goal is to unify existing and new solutions into a standardized ecosystem giving developers the ease and
flexibility to choose the right fit(s) for their projects. Like before, SimpleML does and will not define modeling
algorithms, instead it will focus on perfecting the glue that allows those algorithms and now solutions to be used
in real workflows that can be effortlessly deployed into real applications.

# Architecture
Architecturally, SimpleML has a core set of components that map to the areas of ML management. Each of those in turn
is extended and refined to support external libraries, tools, and infrastructure. Extensibility is the cornerstone for
SimpleML and support for new extensions should be a simple, straightforward process without ever requiring monkey-patching.

Components
- Persistables: Standardization
- Executors: Portability, Scale
- Adapters: Interoperatability
- ORM: Versioning, Lineage, Metadata Tracking, Reusability
- Save Patterns: Persistence
- Registries: Extensibility

SimpleML core acts as the glue code binding all of the components into a
seamless workflow, but all of the components can also be used independently
to leverage portions of the abstraction. See the docs/source code for
detailed instructions on each component or file a request for examples
for a particular implementation.

### Persistables
Persistables are the wrappers around artifacts (artifacts are the actual objects that are generated by training and
need to be deployed into production). They provide a standardized interface to manage and use artifacts making it easy
to use artifacts from different libraries inside the same processing environment. Additionally they allow for a unified
mapping of the particular idiosyncrasies that come with different frameworks to enable developers and scripts to only
use a single access pattern (eg always call "fit" instead of mapping between fit, train, etc based on library).
See the source code for the inheritance pattern and examples to extend around any external library.

### Executors
Executors are the persistable agnostic components that provide portability and scale. They handle execution so the same
functions can be run in various backends without affecting the artifacts produced. (examples: single process execution,
multiprocessing, threading, containers, kubernetes, dask, ray, apache-beam, spark, etc). This intentional decoupling is
a large part of what powers the diverse support for flexible productionalization (train once, deploy anywhere). Note
that not every execution pattern is guaranteed to work natively with every persistable (these will be noted as needed).

### Adapters
Adapters are the complements to persistables and executors. They are optional wrappers to align input requirements
to operations. By definition adapters are stateless wrappers that have no functional impact on processing so they can
be specified at runtime as needed. Additionally the output across different executors for the same operation is
guaranteed to be identical. (eg creating a docker container for a persistable to run in kubernetes or wrapping a persistable in a ParDo to execute in apache-beam)

### ORM
The ORM layer is the heart of metadata management. All persistables are integrated with the database to record
specifications for reproducibility, lineage, and versioning. Depending on the workflows, that metadata can also be
leveraged for reusability to accelerate development iterations by only creating new experiments and reusing old
persistables for existing ones.

### Save Patterns
Save and load patterns are the mechanism that manage persistence. All artifacts can be different with native or special
handling of serialization to save the training state to be loaded into a production environment. Save patterns
allow for that customization to register any serialization/deserialization technique that will automatically be applied
by the persistables. (examples: pickle, hickle, hdf5, json, library native, database tables, etc)

### Registries
Registries are the communication backend that allows users to change internal behavior or extend support at runtime.
Registration can happen implicitly on import or explicitly as part of a script. (eg register serialization class for
a save pattern or map an executor class to a particular backend parameter)


# Workflows
Workflows are largely up to individual developers, but there are some assumptions made about the process:

The primary assumption is that the ML lifecycle follows a DAG. That creates a forward propagating dependency chain
without altering previous pieces of the chain. There is considerable flexibility in what each of the steps
can be, but are generally assumed to flow modularly and mimic a data science project.

Thematic steps, in sequence, start with data management, move through transformation, model creation, and finally evaluation. These are further broken down in the following ways:

Data Management
- Raw Datasets: The basic data block of (potentially) unformatted datasets. These datasets can be sourced from anywhere
- Dataset Pipelines: The required transformation to turn unformatted data into what is expected to be seen in production -- These pipelines are completely optional and only used in derived datasets
- Datasets: The "production formatted" datasets

Transformation
- Pipelines: Transformation sequences to extract and process the dataset

Modeling
- Models: The machine learning models

Evaluation
- Metrics: Evaluation objects computed over the models and datasets


# Examples
Examples will be posted in response to requests under [Examples](https://github.com/eyadgaran/SimpleML/tree/master/examples). Please open an issue on github to request more examples
and tag them with `[Example Request] How to...`

# Usage
Starting a project is as simple as defining the raw data and guiding the transformations. A minimal example using the kaggle Titanic dataset is demonstrated below:

The first step in every project is to establish a database connection to manage metadata. Technically this step is
only necessary if a persistable is saved or loaded, so ephemeral setups can skip this.

```python
from simpleml.utils import Database

# Initialize Database Connection and upgrade the schema to the latest
# By default this will create a new local sqlite database
# The upgrade parameter is only necessary if the db is outdated or new
db = Database().initialize(upgrade=True)
```

The most direct way to use SimpleML is to treat it like other modeling frameworks with forward moving imperative
actions (eg initialize, run methods, save). Notice how this workflow is identical to using the underlying libraries
directly with a few additional parameters. That is because SimpleML wraps the underlying libraries and standardizes
the interfaces.

This block (or any subset) can be executed as many times as desired and will create a new object each time with an
autoincrementing version (for each "name").

```python
from simpleml.constants import TEST_SPLIT
from simpleml.datasets.pandas import PandasFileBasedDataset
from simpleml.metrics import AccuracyMetric
from simpleml.models import SklearnLogisticRegression
from simpleml.pipelines.sklearn import RandomSplitSklearnPipeline
from simpleml.transformers import (
    DataframeToRecords,
    FillWithValue,
    SklearnDictVectorizer,
)


# Create Dataset and save it
dataset = PandasFileBasedDataset(name='titanic',
    filepath='filepath/to/train.csv', format='csv',
    label_columns=['Survived'], squeeze_return=True)
dataset.build_dataframe()
dataset.save()  # this defaults to a pickle serialization

# Define the minimal transformers to fill nulls and one-hot encode text columns
transformers = [
    ('fill_zeros', FillWithValue(values=0.)),
    ('record_coverter', DataframeToRecords()),
    ('vectorizer', SklearnDictVectorizer())
]

# Create Pipeline and save it - Use basic 80-20 test split
# Creates an sklearn.pipelines.Pipeline artifact
pipeline = RandomSplitSklearnPipeline(name='titanic', transformers=transformers,
                               train_size=0.8, validation_size=0.0, test_size=0.2)
pipeline.add_dataset(dataset)  # adds a lineage relationship
pipeline.fit()  # automatically uses relationship and parameters to choose data
pipeline.save()  # this defaults to a pickle serialization

# Create Model and save it -- Creates an sklearn.linear_model.LogisticRegression artifact
model = SklearnLogisticRegression(name='titanic')
model.add_pipeline(pipeline)  # adds a lineage relationship
model.fit()  # automatically uses relationship to choose data
model.save()  # this defaults to a pickle serialization

# Create Metric and save it
metric = AccuracyMetric(dataset_split=TEST_SPLIT)
metric.add_model(model)
metric.add_dataset(dataset)
metric.score()
metric.save()
```

The same operations can also be defined in a declaritive way using wrapper utilities so only the parameters
need to be specified. Additionally if using a deterministic persistable wrapper (the object is fully initialized
at construction and not subject to user changes) the metadata automatically generated can be used to
identify existing artifacts without having to recreate them.

```python
from simpleml.utils import DatasetCreator, MetricCreator, ModelCreator, PipelineCreator

# ---------------------------------------------------------------------------- #
# Option 1: Explicit object creation (pass in dependencies)
# ---------------------------------------------------------------------------- #
# Object defining parameters
dataset_kwargs = {'name': 'titanic', 'registered_name': 'PandasFileBasedDataset',
  'filepath': 'filepath/to/train.csv', 'format': 'csv', 'label_columns': ['Survived'], 'squeeze_return': True}
pipeline_kwargs = {'name': 'titanic', 'registered_name': 'RandomSplitSklearnPipeline', 'transformers': transformers, 'train_size': 0.8, 'validation_size': 0.0, 'test_size': 0.2}
model_kwargs = {'name': 'titanic', 'registered_name': 'SklearnLogisticRegression'}
metric_kwargs = {'registered_name': 'AccuracyMetric', 'dataset_split': TEST_SPLIT}

# each creator has two methods - `retrieve_or_create` and `create`. using create will
# create a new persistable each time while retrieve_or_create will first look for a matching persistable
dataset = DatasetCreator.retrieve_or_create(**dataset_kwargs)
pipeline = PipelineCreator.retrieve_or_create(dataset=dataset, **pipeline_kwargs)
model = ModelCreator.retrieve_or_create(pipeline=pipeline, **model_kwargs)
metric = MetricCreator.retrieve_or_create(model=model, dataset=dataset, **metric_kwargs)

# ---------------------------------------------------------------------------- #
# Option 2: Implicit object creation (pass in dependency references - nested)
# Does not require dependency existence at this time, good for compiling job definitions and executing on remote, distributed nodes
# ---------------------------------------------------------------------------- #
# Nested dependencies
pipeline_kwargs['dataset_kwargs'] = dataset_kwargs
model_kwargs['pipeline_kwargs'] = pipeline_kwargs
metric_kwargs['model_kwargs'] = model_kwargs

# each creator has two methods - `retrieve_or_create` and `create`. using create will
# create a new persistable each time while retrieve_or_create will first look for a matching persistable
dataset = DatasetCreator.retrieve_or_create(**dataset_kwargs)
pipeline = PipelineCreator.retrieve_or_create(dataset_kwargs=dataset_kwargs, **pipeline_kwargs)
model = ModelCreator.retrieve_or_create(pipeline_kwargs=pipeline_kwargs, **model_kwargs)
metric = MetricCreator.retrieve_or_create(model_kwargs=model_kwargs, dataset_kwargs=dataset_kwargs, **metric_kwargs)
```

This workflow is modeled as a DAG, which means that there is room for parallelization, but dependencies are assumed
to exist upon execution. Persistable creators are intentionally designed to take a dependent object as input or
a reference. This allows for job definition before dependencies exist with lazy loading when they are required. Note
that this comes at the cost of additional computations. In order to match up dependencies to a reference, a dummy
persistable must be created and compared, with the exception of a unique reference - like `name, version` which mean
the dependency already exists but is memory efficient to load later. This form also enables usage of config files to
parameterize training instead of requiring an active shell to interactively define the objects.


Once artifacts have been created, they can be easily retrieved by their name attribute (or any other identifying metadata).
By default the latest version for the supplied parameters will be returned, but this can be overridden by explicitly
passing a version number. This makes productionalization as simple as defining a deployment harness to process new
requests.

```python
from simpleml.utils import PersistableLoader

# Notice versions are not shared between persistable types and can increment differently depending on iterations
dataset = PersistableLoader.load_dataset(name='titanic', version=7)
pipeline = PersistableLoader.load_pipeline(name='titanic', version=6)
model = PersistableLoader.load_model(name='titanic', version=8)
metric = PersistableLoader.load_metric(name='classification_accuracy', model_id=model.id)
```

When it comes to production, typically the training data is no longer needed so this mechanism becomes as simple
as loading the feature pipeline and model:

```python
desired_model = PersistableLoader.load_model(name='titanic', version=10)
# Implicitly pass new data through linked pipeline via transform param
desired_model.predict_proba(new_dataframe, transform=True)
```

or (explicitly load a pipeline to use, by default the pipeline the model was trained on will be used)

```python
desired_pipeline = PersistableLoader.load_pipeline(name='titanic', version=11)
desired_model = PersistableLoader.load_model(name='titanic', version=10)
desired_model.predict_proba(desired_pipeline.transform(new_dataframe), transform=False)
```


# The Vision
Ultimately SimpleML should fill the void currently faced by many data scientists with a simple and painless
management layer. Furthermore it will be extended in a way that lowers the technical barrier for all developers
to use machine learning in their projects. If it resonates with you, consider opening a PR and contributing!

Future features I would like to introduce:
- Browser GUI with drag-n-drop components for each step in the process (click on a dataset, pile transformers as blocks, click on a model type...)
- App-Store style tabs for community shared persistables (datasets, transformers...)
- Automatic API hosting for models (click "deploy" for REST API)


# Support
SimpleML is a community project, developed on the side, to address a lot of the pain points I have felt creating ML applications. If you find it helpful and would like to support further development, please consider becoming a [Sponsor :heart:](https://github.com/sponsors/eyadgaran) or opening a PR.


# Contract & Technical Support
For support implementing and extending SimpleML or architecting a machine learning tech stack, contact the author [Elisha Yadgaran](https://www.linkedin.com/in/elishayadgaran/) [:email:](mailto:ElishaY@alum.MIT.edu) for rates.


# Enterprise Support
There is a vision to eventually offer a managed enterprise version, but that is not being pursued at the moment.
Regardless of that outcome, SimpleML will always stay an open source framework and offer a self-hosted version.
