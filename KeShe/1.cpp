#include<iostream>
#include<vector>
using namespace std;
int main()
{
	vector<int> a(45);
	a[0]= 1;
	a[1]= 1;
	for(int i= 2; i<=44; i++){
		a[i]= a[i-1]+a[i-2];
	}
	int n;
	cin>>n;
	cout<<a[n];
	return 0;
}