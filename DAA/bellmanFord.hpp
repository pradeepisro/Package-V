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

struct globalParams{
    int nNodes;
    int nEdges;
    int *index, *edge, *weigth;
    bool *f1, *f2;
};

extern __constant__ globalParams gpuParams;

class bellmanFord{
public:
    bellmanFord(std::string);
    int gpuInit(int);
    void shortestPath(int);
    void displayPrecedence();
private:
    int *h_index, *h_edge, *h_weight;
    int *h_distance, *h_pi;
    bool *h_queue_1, *h_queue_2;
    int *d_index, *d_edge, *d_weight;
    int* d_distance, *d_pi;
    bool *d_queue_1, *d_queue_2;
    std::ifstream fin;
    int nNodes;
    int nEdges;
};

#endif