#include<bits/stdc++.h>
using namespace std;
int mod= 3;
void make(int a, int b){
    int ans= 1;
    a%= mod;
    while(b>0)
	{
		if(b%2==1)  ans=(ans*a)%mod;
		b=b/2;
		a=(a*a)%mod;
	}
    cout<<ans;

}
int main()
{
    make(2, 16);
    return 0;
}