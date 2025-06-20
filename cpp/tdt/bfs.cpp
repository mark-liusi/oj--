#include<iostream>
#include<vector>
#include<queue>
using namespace std;
int search(int m, int n){
    if(m>= n) return m-n;
    int max= 2*n+1;
    vector<int> a(max, 0);
    queue<int> b;
    b.push(m);
    while(!b.empty()){
        int current= b.front(); b.pop();
        for(int i:{current-1, current+1, 2*current}){
            if(i== n){
                a[i]= a[current]+1;
                return a[i];
            }
            if((a[i]!= 0)|| (i<0)|| (a[i]> max)){
                continue;
            }
            b.push(i);a[i]= a[current]+1;
        }
    }
    return 0;
}
int main()
{
    int number;
    cin>>number;
    for(int i= 0; i<number; i++){
        int m, n;
        cin>>m>>n;
        int count= search(m, n);
        cout<<count<<endl;
    }
    return 0;
}