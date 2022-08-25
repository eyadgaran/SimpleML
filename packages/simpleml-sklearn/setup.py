import sys

from setuptools import find_packages, setup

__version__ = "0.14.0"


python_major = sys.version_info.major
python_minor = sys.version_info.minor


# Dependencies have different max versions based on python version
if sys.version_info < (3, 5):  # Python < 3.5
    version_based_dependencies = [
        "scikit-learn<0.21.0",
        "scipy<1.3.0",  # Scikit-learn dependency
    ]
elif sys.version_info < (3, 6):  # Python 3.5
    version_based_dependencies = [
        "scikit-learn<0.23.0",
        "scipy<1.5.0",  # Scikit-learn dependency
    ]
else:
    version_based_dependencies = [
        "scikit-learn",
    ]

# Different extras
test_dependencies = ["coverage"]

setup(
    name="simpleml-sklearn",
    version=__version__,
    description="Simplified Machine Learning",
    author="Elisha Yadgaran",
    author_email="ElishaY@alum.mit.edu",
    license="BSD-3",
    url="https://github.com/eyadgaran/SimpleML",
    download_url="https://github.com/eyadgaran/SimpleML/archive/v{}.tar.gz".format(
        __version__
    ),
    packages=find_packages(),
    include_package_data=True,
    keywords=["machine-learning", "deep-learning", "automated-learning"],
    install_requires=[] + version_based_dependencies,
    extras_require={
        "test": test_dependencies,
    },
    zip_safe=False,
    test_suite="simpleml.tests.load_tests",
    tests_require=test_dependencies,
    entry_points={},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
)
