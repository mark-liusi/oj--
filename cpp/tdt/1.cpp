#include<iostream>
using namespace std;
int main()
{
    int a[30]= {0};
    a[0]= 1;
    a[1]= 1;
    for(int i= 2; i<30; i++){
        a[i]= a[i-1]+a[i-2];
    }
    int number;
    cin>>number;
    for(int i= 0; i<number; i++){
        int order;
        cin>>order;
        cout<<a[order-1]<<endl;
    }
    return 0;
}