#

# Install latest version of CLUEstering and test it
echo "### Installing the latest version of CLUEstering from pip"
pip3 install CLUEstering
echo "## Running the first test"
python3 -m pytest
mv ./file.csv ./file1.csv

# Uninstall CLUE and install the local version of CLUEstering and test it as well
echo "### Unistalling CLUEstering"
yes 2>/dev/null | pip uninstall CLUEstering
echo "### Installing CLUEstering from the repository"
pip install .
echo "## Running the second test"
python3 -m pytest
mv ./file.csv ./file2.csv

# Now we compare the two output files, and they should be identical
echo "### Comparing the outputs of the two versions"
if [[ $(diff file1.csv file2.csv) -eq '' ]]
then 
	echo "## The two outputs are identical, so the test is passed"
	echo "assert(True)" >> test_passed.py
fi
python3 -m pytest
