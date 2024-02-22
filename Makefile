

#Note: Might need to point LD_LIBRARY_PATH to the runtimes after building
CXX=/home/reidatcheson/Applications/clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04/bin/clang++
CXXFLAGS=--std=c++23 --stdlib=libc++ -Wall -pedantic



main : main.cpp sparse.h cg.h
	$(CXX) $(CXXFLAGS) main.cpp -o main



.PHONY : clean


clean :
	rm -rf ./main
