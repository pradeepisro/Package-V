// including CUDA headers...
#include <cuda.h>
#include <cuda_runtime.h>

//standard library header...
#include <iostream>

using namespace std;

class bellmanFord{

public:
    bellmanFord();
private:
    int n;

};

int main(){
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;

    

    return 0;
}