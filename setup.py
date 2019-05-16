from setuptools import setup, find_packages
import sys

__version__ = '0.6'


python_major = sys.version_info.major
python_minor = sys.version_info.minor


# Dependencies have different max versions based on python version
if python_major < 4 and python_minor < 5:  # Python < 3.5
    version_based_dependencies = [
        'scikit-learn<=0.20.3'
    ]
else:
    version_based_dependencies = [
        'scikit-learn'
    ]


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
    keywords=['machine-learning', 'deep-learning', 'automated-learning'],
    install_requires=[
        'sqlalchemy',
        'sqlalchemy_mixins',
        'alembic',
        'pandas',
        'numpy',
        'dill',
        'future',
        'configparser'
    ] + version_based_dependencies,
    extras_require={
        'postgres': ["psycopg2"],
        'deep-learning': ["keras", "tensorflow"],
        'hdf5': ["hickle"],
        'cloud': ["onedrivesdk", "apache-libcloud", "pycrypto"],
        'all': ["psycopg2", "keras", "tensorflow", "hickle", "onedrivesdk", "apache-libcloud"]
    },
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*,>=3.5',
)
