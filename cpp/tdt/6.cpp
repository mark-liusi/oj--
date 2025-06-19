#include<iostream>
#include<queue>
using namespace std;
void search(pair<int, int> a, int y, int &max){
    if((a.first== y)|| (a.second>= max)){
        if(a.second< max){
            max= a.second;
        }
        return;
    }else{
        search({a.first+1, a.second+1}, y, max);
        search({a.first-1, a.second+1}, y, max);
        search({a.first*2, a.second+1}, y, max);
    }
}
int main()
{
    int number;
    cin>>number;
    int max= 1e9;
    for(int i= 0; i<number; i++){
        int x, y; cin>>x>>y;
        pair<int, int> a= {x, 0};
        search(a, y, max);
    }
    cout<<max;
}