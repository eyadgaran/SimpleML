Quickstart
==========

Reading through all the documentation is highly recommended, but for the truly
impatient, following are some quick steps to get started.


Installation
------------

Assuming conda is installed, you can create a new environment as follows (this step is optional,
but recommended to provide a clean working environment)::

    conda create -n simpleml python
    source activate simpleml

Install SimpleML on Python 2.7.x or Python 3.5+ by running the following command::

    pip install simpleml

Refer to the installation guide for other methods of installation (:doc:`Installation<installation>`)


Set Up a Database
-----------------

By default a local sqlite database will be created and used, but Postgres is the
preferred (and tested) database flavor for production systems.
Internally since sqlalchemy is used to manage
all communication, any supported database should work. While it is possible to use SimpleML on
an existing database (provided there are no overlapping tables), it is recommended to
create a new database with the appropriate role-based access.

For example (using postgres), we can create a user `simpleml` and a database `SIMPLEML`::

    CREATE USER simpleml with password 'simpleml';
    CREATE DATABASE SIMPLEML OWNER simpleml;


Create a Project
----------------

Once the database is up and running, you can get started on the modeling task. There
are no restrictions on the project setup, though a juptyer notebook is advised for
prototyping and a python package for deployed services.

Follow on below for an example project in a notebook. Check out the examples repository
for larger implementations.

You can now install jupyter, and any other dependencies via pip::

    pip install jupyter

Starting up the jupyter notebook is as simple as calling::

    jupyter notebook


Train a Model
-------------

From inside the notebook we will conduct a very minimal modeling exercise using
the titanic dataset from kaggle_::

    from simpleml.utils import Database
    from simpleml.datasets import PandasDataset
    from simpleml.pipelines import RandomSplitPipeline
    from simpleml.transformers import SklearnDictVectorizer, DataframeToRecords, FillWithValue
    from simpleml.models import SklearnLogisticRegression
    from simpleml.metrics AccuracyMetric
    from simpleml.constants import TEST_SPLIT


    # Initialize Database Connection - Uses Sqlite Default
    Database().initialize(upgrade=True)

    # Define Dataset and point to loading file
    class TitanicDataset(PandasDataset):
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
    pipeline = RandomSplitPipeline(name='titanic', transformers=transformers,
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
    metric.add_dataset(dataset)
    metric.score()
    metric.save()


Deploy to Production
--------------------

Production models can be hosted pretty much anywhere. We'll just define a basic
API layer using flask and serve predictions from our trained model::

    from flask import Flask, jsonify, request
    import pandas as pd
    from simpleml.utils import PersistableLoader

    # Initialize Database Connection (Same Sqlite DB)
    Database().initialize()

    app = Flask(__name__)
    MODEL = PersistableLoader.load_model(name='titanic', version=1)

    @app.route(/predict, methods=['POST'])
    def predict()
        X = pd.DataFrame(request.json)
        prediction_probability = float(MODEL.predict_proba(X)[:, 1])
        prediction = int(round(prediction_probability, 0))
        return jsonify({'probability': prediction_probability, 'prediction': prediction}), 200


    if __name__ == '__main__':
        app.run()


.. _kaggle: https://www.kaggle.com/c/titanic
