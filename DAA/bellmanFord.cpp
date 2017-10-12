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

vector<int> bellmanFord(graph<int, int> g, int source, int n){
    
    edgeList<int, int> e = graphToEdge(g);

    vector<int> distance(n);
    vector<int> pi(n);
    
    for(int i = 0; i < n; i++){
        distance[i] = INT_MAX;
        pi[i] = -1;
    }

    distance[source] = 0;

    for(int i = 1; i < n; i++){
        for(auto edge : e){
            if(distance[get<1>(edge)] > distance[get<0>(edge)] + get<2>(edge)){
                distance[get<1>(edge)] = distance[get<0>(edge)] + get<2>(edge);
                pi[get<1>(edge)] = get<0>(edge);
            }
        }
    }

    for(auto edge: e){
        if(distance[get<1>(edge)] > distance[get<0>(edge)] + get<2>(edge)){
            cout << "Negative cycle exists.\n";
            cout << "u: " << get<0>(edge) << " " << "v: " << get<1>(edge) << endl;
            break; 
        }
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

    cout << "Bellman Ford:\n";
    auto get_data = bellmanFord(g, 0, n);
    for(int i : get_data){
        cout << i << endl;
    }

    return 0;
}