#include "bellmanFord.hpp"


int main(int argc, char** argv){
    std::string filename =argv[argc - 1];
    bellmanFord b(filename);
    b.shortestPath(0);
    b.displayPrecedence();
    return 0;
}