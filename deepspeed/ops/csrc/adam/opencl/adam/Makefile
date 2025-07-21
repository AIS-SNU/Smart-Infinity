
INC_DIR=includes/
CUDA_DIR=/usr/local/cuda/include/
TARGET=main
OBJECTS=util.o vadd.o

CXXFLAGS=-O3 -Wall -march=native -mavx2 -mfma -fopenmp -mno-avx512f -I$(INC_DIR) -I$(CUDA_DIR)
LDFLAGS=-L/usr/local/cuda/lib64
LDLIBS=-lm -pthread -lOpenCL

CC=g++

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
