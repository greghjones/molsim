INCLUDES := $(shell python3 -m pybind11 --includes; echo -I${CONDA_PREFIX}/include)
EXTENSION := $(shell python3-config --extension-suffix)
LIBNAME := cxx$(EXTENSION) 

$(LIBNAME): functionaltest.cxx
	g++ -O3 -Wall -shared -std=c++23 -undefined dynamic_lookup $(INCLUDES) functionaltest.cxx -o $(LIBNAME)

