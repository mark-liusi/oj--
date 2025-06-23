#include<bits/stdc++.h>
using namespace std;
void judge(vector<int> a){

    for(int i= 0; i<a.size()-1; i++){
        if(a[i]<= a[i+1]){
            continue;
        }else{
            if(a[i]== 1){
                cout<<"-1"<<endl;
                return ;
            }
            
        }
    }
}
int main()
{
    int number;
    cin>>number;
    for(int i= 0; i<number; i++){
        int n;
        cin>>n;
        vector<int> a(n);
        for(int j= 0; j<n; j++){
            cin>>a[j];
        }
        judge(a);
    }
}