#include<iostream>
#include<vector>
#include<algorithm>
#include<queue>
using namespace std;
int solve(vector<int> &c, int cnt, int a, int b){
    int need= 0;
    for(int i: c){
        if(i- cnt*a> 0){
            need+= (i- cnt*a+b-1)/b;
        }
        if(need> cnt){
            return 0;
        }
    }
    return 1;

}
int count(vector<int> &c, int left, int right, int a, int b){
    while(left<right){
        int mid= (left+right)/2;
        if(solve(c, mid, a, b)){
            right= mid;
        }else{
            left= mid+1;
        }
    }
    return left;
}
int main()
{
    int number, a, b;
    cin>>number>>a>>b;
    vector<int> c(number);
    for(int i= 0; i<number; i++){
        cin>>c[i];
    }
    sort(c.begin(), c.end());
    int right= c[number-1]/a;
    cout<<count(c, 0, number, a, b);
    return 0;
}