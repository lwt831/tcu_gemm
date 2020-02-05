CUDA_PATH ?= /usr/local/cuda-10.1
NVCC := $(CUDA_PATH)/bin/nvcc
INCLUDES_CUDA :=$(CUDA_PATH)/include
LIBDIR_CUDA :=$(CUDA_PATH)/lib64
INCLUDES_CUDNN := /usr/local/cuda-10.1/include
LIBDIR_CUDNN := /usr/local/cuda-10.1/lib64



CSRCS := $(shell find . -name '*.cpp' -not -name '._*')
COBJS := $(subst .cpp,.o,$(CSRCS))

CUSRCS := $(shell find . -name '*.cu' -not -name '._*')
CUOBJS := $(subst .cu,.o,$(CUSRCS))

FLAGS = -arch=sm_70 -rdc=true
LIBFILES = -L$(LIBDIR_CUDA) -lcudart -lcuda -lcublas


all: run

run: tallgemm.cu
	#$(NVCC) reduction.cu -arch=sm_70 -rdc=true -L$(LIBDIR_CUDA) -lcudart -lcuda  -I ./cub -o run
	$(NVCC) $(FLAGS) $(LIBFILES) $< -o run

clean:
	#find . -name "*.o" -exec rm -f '{}' ';'

	rm -f run

