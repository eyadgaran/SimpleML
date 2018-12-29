from setuptools import setup, find_packages

__version__ = '0.4'


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
    tests_require=['nose'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)
