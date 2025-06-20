#include<iostream>
#include<stack>
#include<vector>
using namespace std;
int main(){
    int n;
    cin>>n;
    vector<int>a(n);
    for(int i=0;i<n;i++){
        cin>>a[i];  
    }
    stack<int> p;
    int count=0;
    for(int j=0;j<n;j++){
        while(!p.empty()&& p.top()>a[j]){
            p.pop();
            count++;
        }
        p.push(a[j]);
    }
        cout<<count<<endl;
    return 0;
}