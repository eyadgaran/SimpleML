[![PyPI version](https://badge.fury.io/py/simpleml.svg)](https://badge.fury.io/py/simpleml)
![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)
![Docs](https://readthedocs.org/projects/simpleml/badge/?version=stable)
![Build Status](https://travis-ci.org/eyadgaran/SimpleML.svg?branch=master)
![Coverage Status](https://coveralls.io/repos/github/eyadgaran/SimpleML/badge.svg?branch=master)
![HitCount](http://hits.dwyl.io/eyadgaran/simpleml.svg)

# SimpleML
Machine learning that just works, for simple production applications

Documentation: https://simpleml.readthedocs.io/en/latest

Installation:  `pip install simpleml`

# Inspiration
I built SimpleML to simplify some of the most common pain points in new modeling projects. At the heart, SimpleML acts as an abstraction layer to implicitly version, persist, and load training iterations. It is compatible out of the box with popular modeling libraries (Scikit-Learn, Keras, etc) so it shouldn't interfere with normal workflows.

# Architecture
Architecturally, SimpleML should flow modularly and mimic a true data science workflow. The steps, in sequence, start with data management, move through transformation, model creation, and finally evaluation. These are further broken down in the following ways:

Data Management
- Raw Datasets: The basic data block of (potentially) unformatted datasets. These datasets can be sourced from anywhere
- Dataset Pipelines: The required transformation to turn unformatted data into what is expected to be seen in production
- Dataset: The "production formatted" dataset

Transformation
- Pipelines: Transformation sequences to extract and process the dataset

Modeling
- Models: The machine learning models

Evaluation
- Metrics: Evaluation objects computed over the models and the respective dataset


# Usage
There are a few workflows for using SimpleML.

At the core SimpleML defines few constraints allowing for developer flexibility - see extensions SimpleML-Service and SimpleML-Server for blueprints of ready made microservices (just inherit and extend for project specific nuances).

Starting a project is as simple as defining the raw data and guiding the transformations. An example using the kaggle Titanic dataset is demonstrated below:

```python
from simpleml.utils.initialization import Database
from simpleml.datasets.processed_datasets.base_processed_dataset import BaseProcessedDataset
from simpleml.pipelines.production_pipelines.base_production_pipeline import BaseRandomSplitProductionPipeline
from simpleml.transformers.fitful_transformers.vectorizers import SklearnDictVectorizer
from simpleml.transformers.fitless_transformers.converters import DataframeToRecords
from simpleml.transformers.fitful_transformers.fill import FillWithValue
from simpleml.models.classifiers.sklearn.linear_model import SklearnLogisticRegression
from simpleml.metrics.classification import AccuracyMetric
from simpleml.pipelines.validation_split_mixins import TEST_SPLIT


# Initialize Database Connection
db = Database().initialize()

# Define Dataset and point to loading file
class TitanicDataset(BaseProcessedDataset):
    def build_dataframe(self):
        self._external_file = self.load_csv('filepath/to/train.csv')

# Create Dataset and save it
dataset = TitanicDataset(name='titanic', label_columns=['Survived'])
dataset.build_dataframe()
dataset.save()

# Define the minimal transformers to fill nulls and one-hot encode text columns
transformers = [
    ('fill_zeros', FillWithValue(values=0.)),
    ('record_coverter', DataframeToRecords()),
    ('vectorizer', SklearnDictVectorizer())
]

# Create Pipeline and save it - Use basic 80-20 test split
pipeline = BaseRandomSplitProductionPipeline(name='titanic', transformers=transformers,
                                             train_size=0.8, validation_size=0.0, test_size=0.2)
pipeline.add_dataset(dataset)
pipeline.fit()
pipeline.save()

# Create Model and save it
model = SklearnLogisticRegression(name='titanic')
model.add_pipeline(pipeline)
model.fit()
model.save()

# Create Metric and save it
metric = AccuracyMetric(dataset_split=TEST_SPLIT)
metric.add_model(model)
metric.score()
metric.save()
```

This block (or any subset) can be executed as many times as desired and will create a new object each time with an autoincrementing version (for each "name"). Utilities have been defined to share references so that duplication of identical information is not required. Use as follows:

This workflow is modeled as a DAG, which means that there is room for parallelization, but dependencies are assumed to exist upon execution. Persistable creators are intentionally designed to take a dependent object as input or a reference. This allows for job definition before dependencies exist with lazy loading when they are required. Note that this comes at the cost of additional computations. In order to match up dependencies to a reference, a dummy persistable must be created and compared, with the exception of a unique reference - like `name, version` which mean the dependency already exists but is memory efficient to load later.


```python
from simpleml.utils.training.create_persistable import DatasetCreator,\
    PipelineCreator, ModelCreator, MetricCreator

# ---------------------------------------------------------------------------- #
# Option 1: Explicit object creation (pass in dependencies)
# ---------------------------------------------------------------------------- #
# Object defining parameters
dataset_kwargs = {'name': 'titanic', 'registered_name': 'TitanicDataset', 'label_columns': ['Survived']}
pipeline_kwargs = {'name': 'titanic', 'registered_name': 'BaseRandomSplitProductionPipeline', 'transformers': transformers, 'train_size': 0.8, 'validation_size': 0.0, 'test_size': 0.2}
model_kwargs = {'name': 'titanic', 'registered_name': 'SklearnLogisticRegression'}
metric_kwargs = {'registered_name': 'AccuracyMetric', 'dataset_split': TEST_SPLIT}

dataset = DatasetCreator.retrieve_or_create(**dataset_kwargs)
pipeline = PipelineCreator.retrieve_or_create(dataset=dataset, **pipeline_kwargs)
model = ModelCreator.retrieve_or_create(pipeline=pipeline, **model_kwargs)
metric = MetricCreator.retrieve_or_create(model=model, **metric_kwargs)     

# ---------------------------------------------------------------------------- #
# Option 2: Implicit object creation (pass in dependency references - nested)
# Does not require dependency existence at this time, good for compiling job definitions and executing on remote, distributed nodes
# ---------------------------------------------------------------------------- #
# Nested dependencies
pipeline_kwargs['dataset_kwargs'] = dataset_kwargs
model_kwargs['pipeline_kwargs'] = pipeline_kwargs
metric_kwargs['model_kwargs'] = model_kwargs

dataset = DatasetCreator.retrieve_or_create(**dataset_kwargs)
pipeline = PipelineCreator.retrieve_or_create(dataset_kwargs=dataset_kwargs, **pipeline_kwargs)
model = ModelCreator.retrieve_or_create(pipeline_kwargs=pipeline_kwargs, **model_kwargs)
metric = MetricCreator.retrieve_or_create(model_kwargs=model_kwargs, **metric_kwargs)     
```

Once objects have been created, they can be retrieved at whim by their name attribute (with the exception of metrics - which also need reference to the model). By default the latest version for a name will be returned, but this can be overridden by explicitly passing a version number.

```python
from simpleml.utils.scoring.load_persistable import PersistableLoader

# Notice versions are not shared between objects and can increment differently depending on iterations
dataset = PersistableLoader.load_dataset(name='titanic', version=7)
pipeline = PersistableLoader.load_pipeline(name='titanic', version=6)
model = PersistableLoader.load_model(name='titanic', version=8)
metric = PersistableLoader.load_metric(name='classification_accuracy', model_id=model.id)
```

When it comes to production, one typically does not need all the training data so this mechanism becomes as simple as:

```python
desired_model = PersistableLoader.load_model(name='titanic', version=10)
desired_model.predict_proba(new_dataframe)
```


# The Vision
Ultimately SimpleML should fill the void currently faced by many data scientists with a simple and painless persistence layer. Furthermore it will be extended in a way that lowers the technical barrier for all developers to use machine learning in their projects.

Future features I would like to make:
- Browser GUI with drag-n-drop components for each step in the process (click on a dataset, pile transformers as blocks, click on a model type...)
- App-Store style tabs for community sourced objects (datasets, transformers...)
- Native support for transfer learning
- Persistence mechanism for non stable models (traditional models are static - once trained, unaltered), like online learning
- Automatic API hosting for models (click "deploy" for route)


# Support
Help support further development by making a donation to the author

[Donate Now](https://donorbox.org/embed/simpleml?amount=30&show_content=true)


# Contract & Technical Support
For support implementing and extending SimpleML or architecting a machine learning tech stack, contact the author [Elisha Yadgaran](https://www.linkedin.com/in/elishayadgaran/) [:email:](mailto:ElishaY@alum.MIT.edu) for rates.
