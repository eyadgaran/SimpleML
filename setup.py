from setuptools import setup, find_packages

__version__ = '0.3'


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
        'scikit-learn',
        'pandas',
        'numpy',
        'dill',
        'future'
    ],
    extras_require={
        'postgres': ["psycopg2"],
        'deep-learning': ["keras", "tensorflow"],
        'hdf5': ["hickle"],
        'all': ["psycopg2", "keras", "tensorflow", "hickle"]
    },
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose']
)
