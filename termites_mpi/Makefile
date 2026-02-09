MPI_INC = /usr/include/openmpi-x86_64
MPI_LIB = /usr/lib64/openmpi/lib

MPICXX?=g++
CXX?=g++
CXXFLAGS?=-std=c++11 -O3 -fopenmp -I$(MPI_INC)
NVCC?=nvcc
NVFLAGS?=-O2 --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets

# all targets
TARGETS = termites_mpi

# The first rule in the Makefile is the default target that is made if 'make' is invoked with
# no parameters.  'all' is a dummy target that will make everything
default : all

## Dependencies

# all targets depend on the helper programs
$(TARGETS) : termites_mpi.o
LIBS_termites_mpi = -lgomp -lm -ldl -L$(MPI_LIB) -lmpi_cxx -lmpi

CXXFLAGS_termites_mpi =

termites_mpi.o: termites_mpi.cpp termites.h
	$(MPICXX) $(CXXFLAGS) -c termites_mpi.cpp -o termites_mpi.o

termites_mpi: termites_mpi.o
	$(MPICXX) termites_mpi.o $(LIBS_termites_mpi) -o termites_mpi

# wildcard rules
#%MPI.o : %MPI.cpp
#       $(MPICXX) $(CXXFLAGS) $(CFLAGS_$(basename $<)) -c $< -o $@

#%MPI : %MPI.cpp
#       $(MPICXX) $(CXXFLAGS) $(CXXFLAGS_$@) $(filter %.o %.cpp, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@
#%.o : %.cpp
#       $(MPICXX) $(CXXFLAGS) $(CFLAGS_$(basename $<)) -c $< -o $@

% : %.cpp
	$(MPICXX) $(CXXFLAGS) $(CXXFLAGS_$@) $(filter %.o %.cpp, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@

all : $(TARGETS)

clean:
	rm -f $(TARGETS) *.o

.PHONY: clean default all
