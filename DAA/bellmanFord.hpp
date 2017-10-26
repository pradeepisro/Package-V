#ifndef CUDA_HEADER
#define CUDA_HEADER

//host headers...
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>

//device headers...
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <driver_functions.h>

class cudaError_h{
public:
    cudaError_h(cudaError_t err);
};

class bellmanFord{
public:
    bellmanFord(std::string);
    int gpuInit();
    void shortestPath(int source);
    void displayPrecedence();
private:
    int *h_index, *h_edge, *h_weight;
    int *h_distance, *h_pi;
    int *d_index, *d_edge, *d_weight;
    int* d_distance, *d_pi;
    std::ifstream fin;
    int nNodes;
    int nEdges;
};

#endif