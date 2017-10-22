CC=clang++-4.0
CFLAGS= -Wall -g -std=c++14
CUDAFLAGS = -L=/usr/local/cuda/lib64 -lcudart -std=c++14

%: %.cpp
	$(CC) -o $@ $@.cpp $(CFLAGS)

%: %.cu
	$(CC) -o $@ $@.cu $(CUDAFLAGS)

clean:
	find . -maxdepth 1 -type f -executable -exec rm {} +
