# SimpleML
Machine learning that just works, for simple production applications

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
There are a few workflows to using SimpleML.

At the core SimpleML defines few constraints allowing for developer flexibility - see extensions SimpleML-Service and SimpleML-Server for blueprints of ready made microservices (just inherit and extend for project specific nuances).

To start a project is as simple as defining the raw data and guiding the transformations. An example using the kaggle Titanic dataset is demonstrated below:

```python
from simpleml.utils.initialization import Database
from simpleml.datasets.raw_datasets.base_raw_dataset import BaseRawDataset
from simpleml.datasets.processed_datasets.base_processed_dataset import BaseProcessedDataset
from simpleml.pipelines.dataset_pipelines.base_dataset_pipeline import BaseDatasetPipeline
from simpleml.pipelines.production_pipelines.base_production_pipeline import BaseProductionPipeline
from simpleml.transformers.fitful_transformers.vectorizers import SklearnDictVectorizer
from simpleml.transformers.fitless_transformers.converters import DataframeToRecords
from simpleml.transformers.fitful_transformers.fill import FillWithValue
from simpleml.models.classifiers.sklearn.linear_model import SklearnLogisticRegression
from simpleml.metrics.classification import FprTprMetric


# Initialize Database Connection
db = Database(database='titanic').initialize()

# Create Raw Dataset and point to loading file
class TitanicRaw(BaseRawDataset):
    def build_dataframe(self):
        self._dataframe = self.load_csv('filepath/to/train.csv')

raw_dataset = TitanicRaw(name='titanic', label_columns=['Survived'])
raw_dataset.build_dataframe()
raw_dataset.save()

dataset_pipeline = BaseDatasetPipeline(name='titanic')
dataset_pipeline.add_dataset(raw_dataset)
dataset_pipeline.fit()
dataset_pipeline.save()

dataset = BaseProcessedDataset(name='titanic')
dataset.add_pipeline(dataset_pipeline)
dataset.build_dataframe()
dataset.save()

# Define the minimal transformers to fill nulls and one-hot encode text columns
transformers = [
    ('fill_zeros', FillWithValue(values=0.)),
    ('record_coverter', DataframeToRecords()),
    ('vectorizer', SklearnDictVectorizer())
]
pipeline = BaseProductionPipeline(name='titanic', transformers=transformers)
pipeline.add_dataset(dataset)
pipeline.fit()
pipeline.save()

model = SklearnLogisticRegression(name='titanic')
model.add_pipeline(pipeline)
model.fit()
model.save()

metric = FprTprMetric()
metric.add_model(model)
metric.score()
metric.save()
```

This block (or any subset) can be executed as many times as desired and will create a new object each time with an autoincrementing version (for each "name"). Utilities have been defined to share references so that duplication of identical information is not required. Use if so desired.

Once objects have been created, they can be retrieved at whim by their name attribute (with the exception of metrics - which also need reference to the model). By default the latest version for a name will be returned, but this can be overridden by explicitly passing a version number.

ex:
```python
# Initialize Database Connection
db = Database(database='titanic').initialize()

# Notice versions are not shared between objects and can increment differently depending on iterations
raw_dataset = BaseRawDataset.where(name='titanic', version=1)
dataset_pipeline = BaseDatasetPipeline.where(name='titanic', version=3)
dataset = BaseProcessedDataset.where(name='titanic', version=7)
pipeline = BaseProductionPipeline.where(name='titanic', version=6)

model = BaseModel.where(name='titanic', version=8)

# Once references are returned, class properties and external files can be loaded via the load method
raw_dataset.load()
dataset_pipeline.load()
dataset.load()
pipeline.load()
```

When it comes to production, once typically does not need all the training data so this mechanism becomes as simple as:

```python
# Initialize Database Connection
db = Database(database='titanic').initialize()

desired_model = BaseModel.where(name='titanic', version=10).load()

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
