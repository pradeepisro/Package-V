#include "bellmanFord.hpp"

__global__
void relax(int u, int count){
    printf("Relax spawned by thread: %d\n", u);
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid > count){
        return;
    }
    int v = gpuParams.edge[gpuParams.index[u] + tid];
    printf("vertex being processed: %d and distance: %d\n", v, gpuParams.distance[v]);

    if(gpuParams.distance[v] > (gpuParams.distance[u] + gpuParams.weight[gpuParams.index[u] + tid])){
        gpuParams.distance[v] = gpuParams.distance[u] + gpuParams.weight[gpuParams.index[u] + tid];
        gpuParams.f2[v] = true;
        gpuParams.pi[v] = u;
    }

    printf("vertex after processed: %d and distance: %d\n", v, gpuParams.distance[v]);
}