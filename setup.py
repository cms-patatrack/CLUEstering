
import os
import sys
from pathlib import Path
from setuptools import setup

__version__ = "2.2.3"

this_directory = Path(__file__).parent
long_description = (this_directory/'README.md').read_text()

if sys.argv[1] != 'sdist':
    os.system("cmake -B build && make -C build")

setup(
    name="CLUEstering",
    version=__version__,
    author="Simone Balducci",
    author_email="simone.balducci00@gmail.com",
    description='''A library that generalizes the original 2-dimensional CLUE
				algorithm made at CERN.''',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['CLUEstering'],
    install_requires=['scikit-learn','numpy','matplotlib','pandas'],
    package_data={'': []},
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
