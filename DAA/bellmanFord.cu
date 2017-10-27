#include "bellmanFord.hpp"

__constant__ globalParams gpuParams;


cudaError_h::cudaError_h(cudaError_t err){
    if(err != cudaSuccess)
        printf("%s in file %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
}

bellmanFord::bellmanFord(std::string filename){
    fin.open(filename.c_str());
    if(fin.fail() || filename == ""){
        std::cout << "unable to open the file!\n";
        std::exit(0);
    }

    fin >> this->nNodes;
    fin >> this->nEdges;

    this->h_distance = new int[nNodes];
    this->h_pi = new int[nNodes];
    this->h_index = new int[nNodes];
    this->h_queue_1 = new bool[nNodes];
    this->h_queue_2 = new bool[nNodes];

    this->h_edge = new int[nEdges];
    this->h_weight = new int[nEdges];

    int current_node = 0, current_count = 0;
    h_index[0] = 0;
    for(int i = 0, j = 0; i < nNodes; i++){
        fin >> current_node;
        fin >> current_count;
        this->h_queue_1[i] = false;
        this->h_queue_2[i] = false;
        this->h_pi[i] = -1;
        this->h_distance[i] = 100000;
        for(j = 0; j < current_count; j++){
            fin >> h_edge[h_index[current_node] + j];
            fin >> h_weight[h_index[current_node] + j];
        }

        this->h_index[current_node + 1] = h_index[current_node] + current_count;
    }
}

int bellmanFord::gpuInit(int source){

    cudaDeviceProp prop;
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    std::cout << "Number of Devices detected: " << dev_count << std::endl;
    for(int i = 0; i < dev_count; i++){
        cudaError_h(cudaGetDeviceProperties(&prop, 0));
        std::cout << "---------------device detected----------------\n";
        std::cout << "Device name: " << prop.name << std::endl;
        std::cout << "Max Threads per count: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Compute Capability: " << prop.major << std::endl;
        std::cout << "----------------------------------------------\n";
    }

    h_queue_1[source] = true;
    h_distance[source] = 0;

    cudaError_h(cudaMalloc((void**)&this->d_index, this->nNodes * sizeof(int)));
    cudaError_h(cudaMemcpy(this->d_index, this->h_index, this->nNodes * sizeof(int), cudaMemcpyHostToDevice));

    cudaError_h(cudaMalloc((void**)&this->d_distance, nNodes * sizeof(int)));
    cudaError_h(cudaMemcpy(this->d_distance, this->h_distance, nNodes * sizeof(int), cudaMemcpyHostToDevice));

    cudaError_h(cudaMalloc((void**)&this->d_pi, this->nNodes * sizeof(int)));
    cudaError_h(cudaMemcpy(this->d_pi, this->h_pi, this->nNodes * sizeof(int), cudaMemcpyHostToDevice));

    cudaError_h(cudaMalloc((void**)&this->d_edge, this->nEdges * sizeof(int)));
    cudaError_h(cudaMemcpy(this->d_edge, this->h_edge, this->nEdges * sizeof(int), cudaMemcpyHostToDevice));

    cudaError_h(cudaMalloc((void**)&this->d_weight, this->nEdges * sizeof(int)));
    cudaError_h(cudaMemcpy(this->d_weight, this->h_weight, this->nEdges * sizeof(int), cudaMemcpyHostToDevice));

    cudaError_h(cudaMalloc((void**)&this->d_queue_1, this->nEdges * sizeof(bool)));
    cudaError_h(cudaMemcpy(this->d_queue_1, this->h_queue_1, this->nNodes * sizeof(bool), cudaMemcpyHostToDevice));

    cudaError_h(cudaMalloc((void**)&this->d_queue_2, this->nEdges * sizeof(bool)));
    cudaError_h(cudaMemcpy(this->d_queue_2, this->h_queue_2, this->nNodes * sizeof(bool), cudaMemcpyHostToDevice));

    gpuParams.nNodes = nNodes;
    gpuParams.nEdges = nEdges;
    gpuParams.index = d_index;
    gpuParams.edge = d_edge;
    gpuParams.weigth = d_weight;
    gpuParams.f1 = d_queue_1;
    gpuParams.f2 = d_queue_2;

    return 0;
}

void bellmanFord::displayPrecedence(){
    std::cout << "The precedence of the given graph:\n";
    for(int i = 0; i < this->nNodes; i++){
        std::cout << i << " " << this->h_pi[i] << "\n";
    }
}

__device__
void relax(int count){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid > count){
        return;
    }
}

__device__
void computeShortestPath(){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid > gpuParams.nNodes){
        return;
    }

    __shared__ bool toggle;
    toggle = false;

    while(!toggle){
        if(gpuParams.f1[tid]){
            
        }
    }

}

void bellmanFord::shortestPath(int source){
    h_distance[source] = 0;
    this->gpuInit(source);
    //computeShortestPath<<<1, nNodes>>>()
}