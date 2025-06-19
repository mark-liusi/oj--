#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

int main()
{
    int n, m;
    cin>>n>>m;
    vector<int> a(n);
    int sum;
    for(int i= 0; i<n; i++){
        cin>>a[i];
        sum+= a[i];
    }

    int left= 0, right= sum;
    while(left<right){
        int mid= (left+right)/2;
        vector<int> b(m, mid);
        int j= 0;
        int judge= 0;
        for(int i= 0; i<n; i++){
            if(a[i]<= b[j]){
                b[j]-= a[i];
            }else{
                j++;
                i--;
                
            }
            if(j>= m){
                judge= 1;
                break;
            }
        }
        if(judge== 0){
            right= mid;//装得下
        }else{
            left= mid+1;//装不下
        }
    }
    cout<<left;
    return 0;


}
