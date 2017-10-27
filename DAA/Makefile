CC=clang++-4.0
NVCC=nvcc
CFLAGS= -Wall -g -std=c++11
CUDAFLAGS = -g -L/usr/local/cuda/lib64 -lcudart -std=c++14
CUDAINCLUDE = -I/usr/local/cuda/include
THRUST_INCLUDE = -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA -I/usr/local/cuda-8.0/include
INPUT_FILE = "./input/bellmanFord.txt"

.PHONY: clean

%: %.cpp
	$(CC) -o $@ $@.cpp $(CFLAGS)

bellmanFord.o: bellmanFord.cu bellmanFord.hpp
	@ $(CC) -c bellmanFord.cu $(CFLAGS)

%: %.cu
	$(CC) -o $@ $@.cu $(CUDAFLAGS)

main: main.cpp bellmanFord.o
	@ $(CC) -o main main.cpp bellmanFord.o $(CUDAFLAGS) $(CUDAINCLUDE)
	@ ./main $(INPUT_FILE)

final: main bellmanFord.o

clean:
	find . -maxdepth 1 -type f -executable -exec rm {} +
	rm bellmanFord.o
