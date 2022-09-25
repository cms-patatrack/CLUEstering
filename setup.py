from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension
import codecs
import os

__version__ = "1.1.1"

ext_modules = [
	Pybind11Extension("CLUEsteringCPP",
		["CLUEstering/binding.cc"]
	),
]

setup(
    name="CLUEstering",
    version=__version__,
    author="Simone Balducci",
    author_email="simone.balducci00@gmail.com",
    description="A library that generalizes the original 2-dimensional CLUE algorithm made at CERN.",
	 packages=find_packages(),
	 install_requires=['sklearn','numpy','matplotlib','pandas'],
	 keywords=['Python','Clustering','Binding'],
	 classifiers=[
		'Intended Audience :: Developers',
		'Programming Language :: Python :: 3',
		'Operating System :: Unix',
		'Operating System :: MacOS :: MacOS X',
		'Operating System :: Microsoft :: Windows',
	 ]
)
