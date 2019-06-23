.. SimpleML documentation master file, created by
   sphinx-quickstart on Mon Aug  6 22:25:32 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SimpleML
========
Machine learning that just works, for effortless production applications.

It was inspired by common patterns I found myself developing over and over again
for new modeling projects. At the core, it is designed to be minimally intrusive
and provide a clean abstraction for the most common workflows. Each supported
framework details methods to save, version, and subsequently load objects to easily
move from training to production.

SimpleML is designed for data scientists comfortable writing code. Please refer
to the enterprise version (:doc:`Enterprise<enterprise>`) for details on the
extended offering. The enterprise version contains a non-technical interface to
SimpleML as well as additional components to streamline the rest of the machine
learning product workflow.


What It Is
----------
SimpleML is a framework that manages the persistence and tracking of machine
learning objects.


What It Is NOT
--------------
As an abstracted persistence layer, SimpleML does not define any native predictive
algorithms. It wraps existing ones with convenience methods to save, load, and
otherwise manage modeling work.

Prototypical Use Cases:

- deploy locally trained models to remote servers
- define model configs to be trained on a remote server
- experiment with hundreds of different config combinations and track performance

Why use SimpleML over a SAS cloud solution?

- Avoid vendor lockin - fully open source codebase, compatible with any cloud infrastructure and algorithm backend.
- Drop in replacement for most workflows
- Can still deploy your models on-prem without changing your application


Supported Frameworks
--------------------
SimpleML can easily be extended to support almost any modeling framework. These
are the ones that have been developed already:

+--------------+-------------+---------------+--------------------------+
|              | Supervised  | Unsupervised  |  Reinforcement Learning  |
+==============+=============+===============+==========================+
| Scikit-Learn |        X    |               |                          |
+--------------+-------------+---------------+--------------------------+
| Keras        |        X    |               |                          |
+--------------+-------------+---------------+--------------------------+
| PyTorch      |             |               |                          |
+--------------+-------------+---------------+--------------------------+
| Tensorflow   |             |               |                          |
+--------------+-------------+---------------+--------------------------+
| Theano       |             |               |                          |
+--------------+-------------+---------------+--------------------------+
| AI-Gym       |             |               |                          |
+--------------+-------------+---------------+--------------------------+
| Caffe        |             |               |                          |
+--------------+-------------+---------------+--------------------------+
| CNTK         |             |               |                          |
+--------------+-------------+---------------+--------------------------+
| MXNet        |             |               |                          |
+--------------+-------------+---------------+--------------------------+


Source
-----------
You can access the source code at: https://github.com/eyadgaran/SimpleML


Contributing
------------
See guidelines here: Contributing_


Support
-------
SimpleML core is open source and is powered by generous donations. Please donate
if you find it contributing to your projects. Technical support and contract
opportunities are also available - contact the author, `Elisha Yadgaran`_, for details.


Ready to get started? Check out the :doc:`Quickstart<quickstart>` guide.


Index
-----
.. toctree::
   :maxdepth: 1

   Enterprise <enterprise.rst>
   Installation <installation.rst>
   Configuration <configuration.rst>
   Quickstart <quickstart.rst>
   Persistables <persistables.rst>
   Datasets <datasets.rst>
   Pipelines <pipelines.rst>
   Models <models.rst>
   Metrics <metrics.rst>
   Utilities <utilities.rst>
   API <api.rst>
   Changelog <https://github.com/eyadgaran/SimpleML/blob/master/CHANGELOG.md>


.. Links

.. _contributing: https://github.com/eyadgaran/SimpleML/blob/master/CONTRIBUTING.md
.. _`Elisha Yadgaran`: mailto:elishay@alum.mit.edu
