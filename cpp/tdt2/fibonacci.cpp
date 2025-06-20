#include<bits/stdc++.h>
using namespace std;
int main()
{
    int number;
    cin>>number;
    vector<long long> a;
    a.push_back(1);
    a.push_back(1);
    for(long long i= 2; a[i-1]< pow(2, 63); i++){
        a.push_back(a[i-1]+ a[i-2]);
    }
    cout<<a[number+1]%static_cast<long long>(pow(10, 9))+7;
    return 0;
}
