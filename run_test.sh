#

# Install latest version of CLUEstering and test it
pip install CLUEstering
python3 -m pytest
mv ./file.csv ./file1.csv

echo "Finished first phase"

# Uninstall CLUE and install the local version of CLUEstering and test it as well
python3 -m pip uninstall CLUEstering || (echo "assert(False)" >> test_passed.py ; python3 -m pytest)
python3 -m pip uninstall CLUEstering && pip install .
python3 -m pytest
mv ./file.csv ./file2.csv

echo "Finished second phase"

# Now we compare the two output files, and they should be identical
if [[ $(diff file1.csv file2.csv) -eq '' ]]
then 
	echo "assert(True)" >> test_passed.py
fi
python3 -m pytest
