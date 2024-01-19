.PHONY: cc

export CXX := g++
CXXFLAGS = -O3 -std=c++20

all:
	touch run_tests.sh && chmod +x run_tests.sh
	$(CXX) $(CXXFLAGS) ./test_deltaphi.cc -o deltaphi.out
	$(CXX) $(CXXFLAGS) ./test_distance.cc -o distance.out
	$(CXX) $(CXXFLAGS) ./test_kernels.cc -o kernels.out
	echo "#" > run_tests.sh	
	echo ' ' >> run_tests.sh
	echo "./deltaphi.out" >> run_tests.sh
	echo "./distance.out" >> run_tests.sh
	echo "./kernels.out" >> run_tests.sh

deltaphi:
	$(CXX) $(CXXFLAGS) ./test_deltaphi.cc -o deltaphi.out

distance:
	$(CXX) $(CXXFLAGS) ./test_distance.cc -o distance.out

kernels:
	$(CXX) $(CXXFLAGS) ./test_kernels.cc -o kernels.out

clean:
	rm -f ./*.out
	rm -f ./run_tests.sh
	clear

