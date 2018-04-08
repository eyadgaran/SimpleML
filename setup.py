from setuptools import setup, find_packages

setup(
    name='simpleml',
    version='0.1',
    description='Simplified Machine Learning',
    author='Elisha Yadgaran',
    author_email='ElishaY@alum.mit.edu',
    license='BSD-3',
    url='https://github.com/eyadgaran/SimpleML',
    download_url='https://github.com/eyadgaran/SimpleML/archive/0.1.tar.gz',
    packages=find_packages(),
    keywords = ['machine-learning'],
    install_requires=[
        'sqlalchemy',
        'sqlalchemy_mixins',
        'psycopg2',
        'scikit-learn',
        'numpy'
    ],
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose']
)
