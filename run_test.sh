#

# Install latest version of CLUEstering and test it
echo "### Installing the latest version of CLUEstering from pip"
pip3 install CLUEstering
echo "## Running the first test"
START1=$(date +%s)
python3 -m pytest
END1=$(date +%s)
DIFF1=$(( $END - $START ))
mv ./file.csv ./file1.csv

# Uninstall CLUE and install the local version of CLUEstering and test it as well
echo "### Unistalling CLUEstering"
yes 2>/dev/null | pip uninstall CLUEstering
echo "### Installing CLUEstering from the repository"
pip install .
echo "## Running the second test"
START2=$(date +%s)
python3 -m pytest
END2=$(date +%s)
DIFF2=$(( $END - $START ))
mv ./file.csv ./file2.csv

# Now we compare the two output files, and they should be identical
echo "### Comparing the outputs of the two versions"
if [[ $(diff file1.csv file2.csv) -eq '' ]]
then 
	echo "## The two outputs are identical, so the test is passed"
	echo "assert(True)" >> test_output_passed.py
fi

# Now we compare the execution times of the two versions
# The time required by the local version should not be larger than
# that of the old one by more than 20%
echo "### Comparing the execution times of the two versions"
if [[ $(( DIFF2/DIFF1 < 1.2 )) ]]
then 
	echo "## The execution time of the new version is acceptable"
	echo "assert(True)" >> test_timing_passed.py
fi
python3 -m pytest
