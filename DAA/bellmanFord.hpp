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
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <driver_functions.h>

class cudaError_h{
public:
    cudaError_h(cudaError_t err);
};

struct globalParams{
    int nNodes;
    int nEdges;
    int *index, *edge, *weight;
    bool *f1, *f2;
    int *iteration, *distance, *pi;
};

extern __device__ globalParams gpuParams;

class bellmanFord{
public:
    bellmanFord(std::string);
    int gpuInit(int);
    void shortestPath(int);
    void displayPrecedence();
    ~bellmanFord();
private:
    //host vairables...
    int *h_index, *h_edge, *h_weight;
    int *h_distance, *h_pi, *h_iteration;
    bool *h_queue_1, *h_queue_2;

    //device variables...
    int *d_index, *d_edge, *d_weight;
    int *d_distance, *d_pi, *d_iteration;
    bool *d_queue_1, *d_queue_2;

    std::ifstream fin;
    int nNodes;
    int nEdges;
};

extern __global__ void relax(int, int);
extern __global__ void computeShortestPath();
extern __device__ globalParams gpuParams;

#endif