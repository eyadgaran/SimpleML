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
There are a few workflows


# The Vision
Ultimately SimpleML should fill the void currently faced by many data scientists with a simple and painless persistence layer. Furthermore it will be extended in a way that lowers the technical barrier for all developers to use machine learning in their projects.

Future features I would like to make:
- Browser GUI with drag-n-drop components for each step in the process (click on a dataset, pile transformers as blocks, click on a model type...)
- App-Store style tabs for community sourced objects (datasets, transformers...)
- Native support for transfer learning
- Persistence mechanism for non stable models (traditional models are "deterministic" - once trained, unaltered), like online learning
- Automatic API hosting for models (click "deploy" for route)
