#include "bellmanFord.hpp"

__global__
void relax(int u, int count){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid > count){
        return;
    }

}