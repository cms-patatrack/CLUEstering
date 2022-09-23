from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension
import codecs
import os

from distutils import sysconfig
from Cython.Distutils import build_ext

VERSION = '1.0.5'
DESCRIPTION = 'Python library that generalizes the original 2-dimensional CLUE algorithm developed at CERN'

class NoSuffixBuilder(build_ext):
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        suffix = sysconfig.get_config_var('EXT_SUFFIX')
        ext = os.path.splitext(filename)[1]
        return filename.replace(suffix, "") + ext

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

    cmdclass={"build_ext": NoSuffixBuilder},

    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
