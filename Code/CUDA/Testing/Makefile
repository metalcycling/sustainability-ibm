# Unix Shell
SHELL = /bin/bash

# Definitions
DEFINITIONS =

# Libraries
LIB_DIR = /home/metalcycling/Programs

CUDA_INC = -I$(CUDA_HOME)/include
CUDA_LIB = -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcusparse -lcurand -lcuda

# Compiler flags and variables
DEBUG = -O2
CUDA_ARCH = 75

CC = mpicc
CC_FLAGS = -Wall
CC_INCLUDES = -I./

CXX = mpic++
CXX_FLAGS = -Wall -Wno-unused-result
CXX_INCLUDES = -I./ $(CUDA_INC)

CU = nvcc
CU_FLAGS = 
CU_INCLUDES = -I./

FC = mpifort
FC_FLAGS = -fdefault-real-8 -std=legacy
FC_INCLUDES =

LD = nvcc
LD_FLAGS = 
LD_LIBRARIES = $(CUDA_LIB) -ccbin=$(CXX) -gencode arch=compute_$(CUDA_ARCH),"code=sm_$(CUDA_ARCH)" -lstdc++ -lm -lgfortran

# Source files
SRCS_CC := $(shell find . -name '*.c')
SRCS_CXX := $(shell find . -name '*.cpp')
SRCS_CU := $(shell find . -name '*.cu')
SRCS_FC := $(shell find . -name '*.f')
OBJS = $(SRCS_CC:.c=.o) $(SRCS_CXX:.cpp=.o) $(SRCS_FC:.f=.o) $(SRCS_CU:.cu=.o)

# Executable
EXE = main

# Make rules
all: $(EXE)

$(EXE): $(OBJS)
	$(LD) $(DEBUG) $(DEFINITIONS) $(LD_FLAGS) -o $@ $(OBJS) $(LD_LIBRARIES)

%.o: %.c
	$(CC) $(DEBUG) $(DEFINITIONS) $(CC_FLAGS) $(CC_INCLUDES) -c $< -o $@

%.o: %.cu
	$(CU) $(DEBUG) $(DEFINITIONS) $(CU_FLAGS) $(CU_INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(DEBUG) $(DEFINITIONS) $(CXX_FLAGS) $(CXX_INCLUDES) -c $< -o $@

%.o: %.f
	$(FC) $(DEBUG) $(DEFINITIONS) $(FC_FLAGS) $(FC_INCLUDES) -c $< -o $@

clean:
	rm *.o $(EXE);
