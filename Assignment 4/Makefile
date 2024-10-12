CXX                = mpicxx -std=c++20
GXX                = g++ -std=c++20
OPTFLAGS           = -O3
CXXFLAGS           = -Wall
OPENMP             = -fopenmp
LIBS               = 
SOURCES            = $(wildcard *.cpp)
TARGETS            = $(patsubst %.cpp,%,$(SOURCES))

.PHONY: all clean cleanall 

all: nkeyspar nkeys $(filter-out nkeyspar nkeys, $(TARGETS))

nkeyspar: nkeyspar.cpp
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ $< $(LIBS)

nkeyspar-old: nkeyspar-old.cpp
	$(CXX) $(CXXFLAGS) $(OPENMP) $(OPTFLAGS) -o $@ $< $(LIBS)

nkeys: nkeys.cpp
	$(GXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ $< $(LIBS)

clean: 
	-rm -f *.o *~
cleanall: clean
	-rm -f $(TARGETS)

