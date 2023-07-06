#

# Install latest version of CLUEstering and test it
echo "### Installing the latest version of CLUEstering from pip"
pip3 install CLUEstering==1.4.0
echo "## Running the first test"
START1=$(date +%s)
python3 -m pytest test_versions.py
END1=$(date +%s)
DIFF1=$(( $END1 - $START1 )).0
mv ./file.csv ./file1.csv

# Uninstall CLUE and install the local version of CLUEstering and test it as well
echo "### Unistalling CLUEstering"
yes 2>/dev/null | pip uninstall CLUEstering
echo "### Installing CLUEstering from the repository"
cd ..
pip install .
cd tests
echo "## Running the second test"
START2=$(date +%s)
python3 -m pytest test_versions.py
END2=$(date +%s)
DIFF2=$(( $END2 - $START2 )).0
mv ./file.csv ./file2.csv

# The local version is still installed
# Execute the test for the different input data types
echo "## Testing that CLUE works for all the supported data types"
python3 -m pytest test_input_datatypes.py

# Now we compare the two output files, and they should be identical
echo "### Comparing the outputs of the two versions"
if [[ $(diff file1.csv file2.csv) -eq '' ]]
then 
	echo "## The two outputs are identical, so the test is passed"
	echo "def test(): assert(True)" > test_output_passed.py
fi
python3 -m pytest test_output_passed.py

# Run the tests of the test datasets
python3 -m pytest test_*_dataset.py
