#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <climits>
#include <queue>

using namespace std;


template<typename T, typename U>
using graph = map<T, vector<pair<T, U>>>;

template<typename T, typename U>
using edgeList = vector<tuple<T, T, U>>;


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

vector<int> frontier(graph<int, int> g, int source, int n){

    vector<int> distance(n);
    vector<int> pi(n);

    for(int i = 0; i < n; i++){
        distance[i] = 100000000;
        pi[i] = -1;
    }

    distance[source] = 0;

    priority_queue<int> f1, f2;
    f1.push(source);

    while(!f1.empty()){
        int size = f1.size();
        for(int i = 0; i < size; i++){
            int u = f1.top();
            f1.pop();
            for(auto j : g[u]){
                int v = get<0>(j), w = get<1>(j);
                if(distance[v] > (distance[u] + w)){
                    distance[v] = distance[u] + w;
                    pi[v] = u;
                    f2.push(v);
                }
            }
        }
        f2.swap(f1);
    }

    return pi;
}

int main(){
    fstream file("./bellmanFord.txt");
    if(file.fail()){
        cout << "File unable to open\n";
        return 0;
    }

    // ##This is the adjacency list...
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

    //calling statement goes here...
    cout << "frontier algorithm:\n";
    auto get_data = frontier(g, 0, n);
    for(auto i : get_data){
        cout << i << endl;
    }
    return 0;
}