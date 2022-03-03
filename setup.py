import sys

from setuptools import find_packages, setup

__version__ = '0.12.0'


python_major = sys.version_info.major
python_minor = sys.version_info.minor


# Dependencies have different max versions based on python version
if sys.version_info < (3, 5):  # Python < 3.5
    version_based_dependencies = [
        'scikit-learn<0.21.0',
        'scipy<1.3.0',  # Scikit-learn dependency
        'pandas<0.25.0',
    ]
elif sys.version_info < (3, 6):  # Python 3.5
    version_based_dependencies = [
        'scikit-learn<0.23.0',
        'scipy<1.5.0',  # Scikit-learn dependency
        'pandas<1.0.0',
        'markupsafe<2.0.0',
    ]
elif sys.version_info <= (3, 6, 1):  # Python 3.6
    version_based_dependencies = [
        'scikit-learn',
        'pandas<1.0.0',
    ]
else:
    version_based_dependencies = [
        'scikit-learn',
        'pandas',
    ]

# Different extras
postgres_dependencies = ["psycopg2"]
deep_learning_dependencies = ["tensorflow>=2", "hickle"]
cloud_dependencies = ["apache-libcloud", "pycrypto", "sshtunnel"]
onedrive_dependencies = ["onedrivesdk<2"]  # Python support EOL >2
dask_dependencies = ["dask[complete]", "pyarrow"]
test_dependencies = ["coverage"]
all_dependencies = list(set(
    postgres_dependencies
    + deep_learning_dependencies
    + cloud_dependencies
    + onedrive_dependencies
    + dask_dependencies
))
test_dependencies = all_dependencies + test_dependencies

setup(
    name='simpleml',
    version=__version__,
    description='Simplified Machine Learning',
    author='Elisha Yadgaran',
    author_email='ElishaY@alum.mit.edu',
    license='BSD-3',
    url='https://github.com/eyadgaran/SimpleML',
    download_url='https://github.com/eyadgaran/SimpleML/archive/v{}.tar.gz'.format(__version__),
    packages=find_packages(),
    include_package_data=True,
    keywords=['machine-learning', 'deep-learning', 'automated-learning'],
    install_requires=[
        'sqlalchemy>=1.3.7',  # Unified json_serializer/deserializer for sqlite
        'sqlalchemy-mixins',
        'sqlalchemy-json',
        'alembic',
        'numpy',
        'cloudpickle',
        'future',
        'configparser',
        'simplejson',
        'click',
    ] + version_based_dependencies,
    extras_require={
        'postgres': postgres_dependencies,
        'deep-learning': deep_learning_dependencies,
        'cloud': cloud_dependencies,
        'onedrive': onedrive_dependencies,
        'dask': dask_dependencies,
        'all': all_dependencies,
        'test': test_dependencies,
    },
    zip_safe=False,
    test_suite='simpleml.tests.load_tests',
    tests_require=test_dependencies,
    entry_points={
        'console_scripts': [
            'simpleml=simpleml.cli.main:cli',
            'simpleml-test=simpleml.tests:run_tests',
            'simpleml-unit-test=simpleml.tests.unit:run_tests',
            'simpleml-integration-test=simpleml.tests.integration:run_tests',
            'simpleml-regression-test=simpleml.tests.regression:run_tests',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
)
