# CUDA installation directory
CUDA_INSTALL_PATH := /usr/local/cuda-11.7/

# cuBLAS installation directory
CUBLAS_INSTALL_PATH := /usr/local/cuda-11.7/

# Compiler
NVCC := $(CUDA_INSTALL_PATH)/bin/nvcc

# Compiler flags
NVCCFLAGS := -arch=sm_86
CFLAGS := -I$(CUDA_INSTALL_PATH)/include -I$(CUBLAS_INSTALL_PATH)/include
LDFLAGS := -L$(CUDA_INSTALL_PATH)/lib64 -L$(CUBLAS_INSTALL_PATH)/lib64 -lcublas

# Target executable
TARGET := cublas_gemm_batch

# Source files
SRC := cublas_gemmBatched_example.cu 

# Object files
OBJ := $(SRC:.cu=.o)

# Rules
all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) $(LDFLAGS) $^ -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ)