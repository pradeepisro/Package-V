#include "bellmanFord.hpp"

__global__
void relax(int u, int count){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int v = gpuParams.edge[gpuParams.index[u] + tid];
    if(tid > count || v == u){
        return;
    }
    printf("Relax spawned by thread: %d\n", u);
    printf("vertex being processed: %d and distance: %d\n", v, gpuParams.distance[v]);

    if(gpuParams.distance[v] > (gpuParams.distance[u] + gpuParams.weight[gpuParams.index[u] + tid])){
        gpuParams.distance[v] = gpuParams.distance[u] + gpuParams.weight[gpuParams.index[u] + tid];
        if(gpuParams.index[v + 1] - gpuParams.index[v])
            gpuParams.f2[v] = true;
        gpuParams.pi[v] = u;
    }

    printf("vertex after processed: %d and distance: %d\n", v, gpuParams.distance[v]);
}