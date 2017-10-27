#include "bellmanFord.hpp"


int main(int argc, char** argv){
    std::string filename;
    if(argc < 3)
        filename = argv[argc - 1];
    else
        filename = "";
    bellmanFord b(filename);
    b.shortestPath(0);
    b.displayPrecedence();
    return 0;
}