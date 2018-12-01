from setuptools import setup, find_packages
from .simpleml._version import __version__


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
    keywords = ['machine-learning'],
    install_requires=[
        'sqlalchemy',
        'sqlalchemy_mixins',
        'psycopg2',
        'scikit-learn',
        'numpy',
        'dill',
        'hickle',
        'keras',
        'pandas'
    ],
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose']
)
