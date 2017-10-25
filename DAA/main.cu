// including CUDA headers...
#include <cuda.h>
#include <cuda_runtime.h>

//standard library header...
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <tuple>
#include <algorithm>
#include <set>

using namespace std;

template<typename T, typename U>
using graph = map<int, vector<pair<T, U>>>;

template<typename T, typename U>
using edgeList = vector<tuple<T, T, U>>;

int *index_arr, *edge_arr, *weight_arr, *d_out;
int *l_index_arry, *l_edge_arr, *l_weight_arr;
int *distance, *pi;
bool *f1, *f2;

edgeList<int, int> graphToEdge(graph<int, int> g){
    map<pair<int, int>, int> temp;
    set<set<int>> s;

    for(auto keys : g){
        for(auto i : keys.second){
            if(s.find({keys.first, i.first}) == s.end()){
                s.insert({keys.first, i.first});
                temp[{keys.first, i.first}] = i.second;
            }
        }
    }

    edgeList<int, int> e;

    for(auto keys: temp){
        e.push_back({keys.first.first, keys.first.second, keys.second});
    }

    return e;
}

__device__
void relax(int u, int v, int w, int n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx > n){
        return 0;
    }

    

}

__global__
void bellmanFord(const int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n){
        return;
    }

    __shared__ bool toggle = false;
    int count = 0;

    while(!toggle){
        if(tid == 0){
            toggle = true;
        }
        if(f1[idx]){
            if(idx > n - 1){
                count = index_arr[idx + 1] - index_arr[idx];
            } 
        }
    }
}

int graphInit(graph<int, int> g, int n){

    cudaMalloc((void**)&f1, n * sizeof(bool));
    cudaMalloc((void**)&f2, n * sizeof(bool));

    cudaMalloc((void**)&l_index_arry, sizeof(int));
    cudaMemset((void*)l_index_arry, n, sizeof(int));

    cudaMalloc((void**)&distance, n * sizeof(int));
    cudaMemset((void*)distance[0], 0, sizeof(int));
    cudaMemset((void*)pi[i], -1, sizeof(int));

    for(int i = 1; i < n; i++){
        cudaMemset((void*)distance[i], 0, sizeof(int));
        cudaMemset((void*)pi[i], -1, sizeof(int));
    }

    int *h_index_arr = new int[n]{0, 2, 2, 8, 10, 12, 12, 14, 15};
    cudaMalloc((void**)&index_arr, n * sizeof(int));

    edgeList<int, int> e = graphToEdge(g);

    // sort(e.begin(), e.end(), 
    // [](auto i, auto j){
    //     return get<0>(i) < get<0>(j);
    // });

    int n_edges = e.size();

    int *h_edge_arr = new int[n_edges]{2, 3, 4, 2, 8, 1, 3, 8, 7, 5, 4, 4, 3, 6}
    , *h_weight_arr = new int[n_edges]{1, 3, 10, 0, 2, 2, 1, 1, 2, 4, 0, 3, 3, 1};

    cudaMalloc((void**)&l_edge_arr, sizeof(int));
    cudaMemset((void*)l_edge_arr, n_edges, sizeof(int));

    cudaMalloc((void**)&l_weight_arr, sizeof(int));
    cudaMemset((void*)l_weight_arr, n_edges, sizeof(int));

    cudaMalloc((void**)&edge_arr, n_edges * sizeof(int));
    cudaMalloc((void**)&weight_arr, n_edges * sizeof(int));

    cudaMemcpy(index_arr, h_index_arr, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_arr, h_weight_arr, n_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(edge_arr, h_edge_arr, n_edges * sizeof(int), cudaMemcpyHostToDevice);

    return 0;
}

int main(){
    
    fstream file("./input/bellmanFord.txt");
    if(file.fail()){
        cout << "Unable to open the file!\n";
        return -1;
    } 

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;


    cout << "---------------Device Properties---------------------" << endl;
    cout << "Device name: " << prop.name << endl;
    cout << "Max Threads per block: " << prop.maxThreadsPerBlock << endl;
    cout << "Compute Capability: " << prop.major << endl;
    cout << "------------------------------------------------------" << endl;


    graph<int, int> g;
    
    int n = 0;
    file >> n;
    
    for(int i = 0; i < n; i++){
        int v = 0;
        file >> v;

        int m = 0;
        file >> m;
        
        int node = 0, weight = 0;
        for(int j = 0; j < m; j++){
            file >> node >> weight;
            g[v].push_back({node, weight});
        }
    }
    


    graphInit(g, n);

    bellmanFord<<<1, n>>>(n);

    return 0;
}