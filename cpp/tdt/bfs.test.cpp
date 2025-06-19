#include<iostream>
#include<queue>
#include<vector>
using namespace std;
int search(int x, int y){
    if(y<= x) return x-y;
    int max= 2*y+1;
    vector<int> a(max, 0);
    queue<int> b;
    b.push(x);
    while(!b.empty()){
        int current= b.front(); b.pop();
        if(current== y){
            return a[current];
        }
        for(int i:{current-1, current+1, current*2 }){
            if((i<0)|| i>max){
                continue;
            }else if(a[i]!= 0){
                continue;
            }else{
                a[i]= a[current]+1;
                b.push(i);
            }
        }
    }
    return -1;
}
int main()
{
    int number;
    cin>>number;
    for(int i= 0; i<number; i++){
        int x, y;
        cin>>x>>y;
        int count= search(x, y);
        cout<<count<<endl;
    }
    return 0;
}