from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension
import codecs
import os

VERSION = '0.0.1'
DESCRIPTION = 'Python library that generalizes the original 2-dimensional CLUE algorithm developed at CERN'

ext_modules = [
	Pybind11Extension("clusteringAlgo",
		["CLUEstering/binding.cc"]
	),
]

# Setting up
setup(
    name="CLUEstering",
    version=VERSION,
    author="Simone Balducci",
    author_email="<simone.balducci00@gmail.com>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=[],
	 ext_modules=ext_modules,
    keywords=['Python', 'Clustering', 'Binding'],
    classifiers=[
        "Development Status :: Production",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
