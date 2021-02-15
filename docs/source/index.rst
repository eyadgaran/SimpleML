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
framework details methods to save, version, and subsequently load artifacts to easily
move from training to production.

SimpleML is designed for data scientists comfortable writing code. Please refer
to the enterprise version (:doc:`Enterprise<enterprise>`) for details on the
extended offering. The enterprise version contains a non-technical interface to
SimpleML as well as additional components to streamline the rest of the machine
learning product workflow.


What It Is
----------
SimpleML is a metadata and persistence framework for Machine Learning projects.
The primary focus is on enabling a seamless developer experience with tracking,
reproducibility, and deployment of an ML model or service into a production
environment. The emphasis on seamless means architecting from the ground up
to incorporate with minimal overhead and drop-ins for leveraging existing
libraries and workflows. For advanced users and extreme customizability,
SimpleML is designed with extensibility as a first-class concern with carefully
designed registries, class hierarchies, and mixins.


What It Is NOT
--------------
As an abstracted persistence layer, SimpleML does not define any native predictive
algorithms. It wraps existing ones with convenience methods to save, load, and
otherwise manage modeling work.

The choice to focus on a low overhead setup (automatic local database or
configured central instance and cloud storage tokens) means that SimpleML
is best suited for an individual developer ("single laptop") or small-medium
data science team ("single cluster"). Bigger teams and organizations can
still use SimpleML, and may very well benefit from it, but likely are in a
position to invest in an enterprise ML management system with a dedicated
support team and infrastructure.

Typical Use Cases:

- deploy locally trained models to remote servers
- define model configs to be trained on a remote server
- experiment with hundreds of different config combinations and track performance

Why use SimpleML over a SAS cloud solution?

- Avoid vendor/environment lock-in. Retain full control over resources and results
  - Fully open source codebase
  - Compatible with any cloud or on-prem compute and hosting infrastructure
  - Deployable through a variety of mediums (microservices, containers, serverless, raw binaries)
  - Extensible and compatible with almost every framework and algorithm backend
- Drop in replacement for most workflows
- Thinly wraps popular open-source libraries to constantly stay up to date with latest versions
- Easily extensible to internal and private source libraries


Reasons NOT to use SimpleML
---------------------------
While there aren't many reasons to not use SimpleML in most development environments,
the following are some known limitations that will impact the overall value derived
from SimpleML:

- Big data (truly big data). The underlying dataset and processing components can be swapped with
big data centric ones, but the defaults are targeting small-medium sized data.

- Distributed environments
Running highly distributed code and data processing requires expensive infrastructure and
support. Projects at that scale will likely already have the resources and appetite
to invest in and manage other enterprise solutions.

Supported Frameworks
--------------------
SimpleML can easily be extended to support almost any modeling framework. These
are the ones that have been developed already:

+--------------------+-------------+---------------+--------------------------+
|                    | Supervised  | Unsupervised  |  Reinforcement Learning  |
+====================+=============+===============+==========================+
| Scikit-Learn       |        X    |               |                          |
+--------------------+-------------+---------------+--------------------------+
| Tensorflow (Keras) |        X    |               |                          |
+--------------------+-------------+---------------+--------------------------+
| PyTorch            |             |               |                          |
+--------------------+-------------+---------------+--------------------------+
| Theano             |             |               |                          |
+--------------------+-------------+---------------+--------------------------+
| AI-Gym             |             |               |                          |
+--------------------+-------------+---------------+--------------------------+
| Caffe              |             |               |                          |
+--------------------+-------------+---------------+--------------------------+
| CNTK               |             |               |                          |
+--------------------+-------------+---------------+--------------------------+
| MXNet              |             |               |                          |
+--------------------+-------------+---------------+--------------------------+


Source
-----------
Source code can be accessed on GitHub: https://github.com/eyadgaran/SimpleML


Contributing
------------
See guidelines here: Contributing_


Support
-------
SimpleML core is open source and powered by generous donations. Please donate
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
   Database <database.rst>
   Persistables <persistables.rst>
   Datasets <datasets.rst>
   Pipelines <pipelines.rst>
   Models <models.rst>
   Metrics <metrics.rst>
   Structs <structs.rst>
   CLI <cli.rst>
   Utilities <utilities.rst>
   API Reference <api/index.rst>
   Changelog <https://github.com/eyadgaran/SimpleML/blob/master/CHANGELOG.md>


.. Links

.. _contributing: https://github.com/eyadgaran/SimpleML/blob/master/CONTRIBUTING.md
.. _`Elisha Yadgaran`: mailto:elishay@alum.mit.edu
