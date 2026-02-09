CUDA_INC = /opt/local/stow/cuda-11.1/include
CUDA_LIB = /opt/local/stow/cuda-11.1/lib64

NVCC?=nvcc
NVFLAGS?= -O3 --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets -G -lineinfo 

TARGETS = termites_gpu

default : all

%.o : %.cu
	$(NVCC) $(NVFLAGS) $(NVFLAGS_$(basename $<)) -c $< -o $@

% : %.cu
	$(NVCC) $(NVFLAGS) $(NVFLAGS_$@) $(filter %.o %.cu, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@

all : $(TARGETS)

clean:
	rm -f $(TARGETS) *.o

.PHONY: clean default all
