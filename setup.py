from setuptools import setup

__version__ = "1.1.1"

setup(
    name="CLUEstering",
    version=__version__,
    author="Simone Balducci",
    author_email="simone.balducci00@gmail.com",
    description="A library that generalizes the original 2-dimensional CLUE algorithm made at CERN.",
	 install_requires=['sklearn','numpy','matplotlib','pandas']
)
