from pathlib import Path
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension

__version__ = "1.4.0"
this_directory = Path(__file__).parent
long_description = (this_directory/'README.md').read_text()

ext_modules = [
	Pybind11Extension(
		"CLUEsteringCPP",
		['CLUEstering/binding.cc'],
        include_dirs = ['CLUEstering/include/']
	),
]

setup(
    name="CLUEstering",
    version=__version__,
    author="Simone Balducci",
    author_email="simone.balducci00@gmail.com",
    description='''A library that generalizes the original 2-dimensional CLUE
				 algorithm made at CERN.''',
	 long_description=long_description,
	 long_description_content_type='text/markdown',
	 packages=find_packages(),
	 install_requires=['scikit-learn','numpy','matplotlib','pandas'],
	 ext_modules=ext_modules,
	 keywords=['Python','Clustering','Binding'],
	 python_requires='>=3.7',
	 classifiers=[
		'Intended Audience :: Developers',
		'Programming Language :: Python :: 3',
		'Operating System :: Unix',
		'Operating System :: MacOS :: MacOS X',
		'Operating System :: Microsoft :: Windows',
	 ]
)
