#include <iostream>
#include <set>

using namespace std;

int main(){
    set<int> u, v;
    u.insert(1);
    u.insert(2);
    v.insert(2);
    v.insert(1);

    if(u == v){
        cout << "They are same\n";
    }
    return 0;
}