CC=clang++
NVCC=nvcc
CFLAGS= -Wall -m64 -O3 -std=c++14 -g
NVCCFLAGS = -std=c++11 -code=compute_35
SFLAGS = -arch=compute_35 -rdc=true
CUDAFLAGS = -L/usr/local/cuda/lib64 -lcudadevrt -lcudart -std=c++14
CUDAINCLUDE = -I/usr/local/cuda/include
THRUST_INCLUDE = -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -I/usr/local/cuda-8.0/include
INPUT_FILE = "./input/bellmanFord.txt"

.PHONY: clean all

all: main
	./main $(INPUT_FILE)

%: %.cpp
	$(CC) -o $@ $@.cpp $(CFLAGS)

relax.o: relax.cu bellmanFord.hpp
	$(NVCC) $(SFLAGS) -c relax.cu

bellmanFord.o: bellmanFord.cu bellmanFord.hpp
	$(NVCC) $(SFLAGS) -c bellmanFord.cu

%: %.cu
	$(CC) -o $@ $@.cu $(CUDAFLAGS)

link.o: relax.o bellmanFord.o
	$(NVCC) $(SFLAGS) --device-link relax.o bellmanFord.o --output-file link.o

main.o: main.cpp
	$(CC) -c main.cpp $(CFLAGS) $(CUDAINCLUDE)

main: main.o relax.o bellmanFord.o link.o
	$(CC) $(CFLAGS) main.o link.o relax.o bellmanFord.o -o main $(CUDAFLAGS)

final: main bellmanFord.o

clean:
	find . -maxdepth 1 -type f -executable -exec rm {} +
	rm bellmanFord.o relax.o link.o main.o
