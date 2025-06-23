#include<bits/stdc++.h>
using namespace std;
long long mod= 1e9+ 7;
void matrix(long long (&a)[2][2], long long b[2][2]){
    long long temp[2][2]= {{0, 0}, {0, 0}};
    temp[0][0]= (a[0][0]*b[0][0]+a[0][1]*b[1][0])%mod;
    temp[0][1]= (a[0][0]*b[0][1]+a[0][1]*b[1][1])%mod;
    temp[1][0]= (a[1][0]*b[0][0]+a[1][1]*b[1][0])%mod;
    temp[1][1]= (a[1][0]*b[0][1]+a[1][1]*b[1][1])%mod;
    for(int i= 0; i<2; i++){
        for(int j= 0; j<2; j++){
            a[i][j]= temp[i][j];
        }
    }
}
void make(long long number){
    if (number == 0) { cout << 0 << '\n'; return; }
    if (number == 1) { cout << 1 << '\n'; return; }
    long long a[2][2]= {{1,1}, {1, 0}};
    long long end[2][2]= {{1, 0}, {0, 1}};
    number-= 1;
    while(number> 0){
        if(number%2== 1) matrix(end, a);
        number= number>>1;
        matrix(a, a);
    }
    cout<<end[0][0];
}
int main()
{
    long long number;
    cin>>number;
    make(number);
    return 0;
}
