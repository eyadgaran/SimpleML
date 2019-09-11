Installation
============

SimpleML currently runs best on Python 2.7.x and 3.5+; other versions may work
but are not explicitly supported.

You can install SimpleML via several different methods. The simplest is via
`pip <http://www.pip-installer.org/>`_::

    pip install simpleml


While the above is the simplest method, the recommended approach is to create a
virtual environment via conda, before installation.
Assuming you have conda installed, you can then open a new terminal
session and create a new virtual environment::

    conda create -n simpleml python
    source activate simpleml

Once the virtual environment has been created and activated, SimpleML can be
installed via ``pip install simpleml`` as noted above. Alternatively, if you
have the project source, you can install SimpleML using the distutils method::

    cd path-to-source
    python setup.py install

If you have Git installed and prefer to install the latest bleeding-edge
version rather than a stable release, use the following command::

    pip install -e "git+https://github.com/eyadgaran/simpleml.git@master#egg=simpleml"


Optional packages
-----------------

SimpleML comes configured with a few optional dependencies. The base installation
will work without any issues if they are not installed, but extended functionality
will not be available.

The general command is ``pip install simpleml[extras]`` (substituting "extras" with the group of dependencies)

These are the current supported extras::

    'postgres': ["psycopg2"]
    'deep-learning': ["keras", "tensorflow"]
    'hdf5': ["hickle"]
    'cloud': ["onedrivesdk", "apache-libcloud", "pycrypto"]

Additionally, a convenience extra titled ``all`` is defined to install the full list
of optional dependencies.


Dependencies
------------

When SimpleML is installed, the following dependent Python packages should be
automatically installed. These are necessary dependencies and SimpleML will not
function properly without them.

* `sqlalchemy <http://pypi.python.org/pypi/sqlalchemy>`_, to manage database communication
* `sqlalchemy_mixins <http://pypi.python.org/pypi/sqlalchemy_mixins>`_, for database active record functionality
* `alembic <http://pypi.python.org/pypi/alembic>`_, for database schema management
* `pandas <http://pypi.python.org/pypi/pandas>`_, for data processing
* `numpy <http://pypi.python.org/pypi/numpy>`_, for numerical foundations
* `cloudpickle <http://pypi.python.org/pypi/cloudpickle>`_, for code pickling
* `future <http://pypi.python.org/pypi/future>`_,  for Python 2 and 3 compatibility
* `configparser <http://pypi.python.org/pypi/configparser>`_, for config management
* `simplejson <https://pypi.python.org/pypi/simplejson>`_, extended support for json handling
* `scikit-learn <https://pypi.org/project/scikit-learn>`_, base machine learning support and metric computation


Upgrading
---------

If you installed SimpleML via ``pip`` and wish to upgrade to
the latest stable release, you can do so by adding ``--upgrade``::

    pip install --upgrade simpleml

If you installed via distutils or the bleeding-edge method, simply
perform the same step to install the most recent version.
