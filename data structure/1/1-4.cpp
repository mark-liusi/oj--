#include<iostream>
#include<string>
using namespace std;
int main()
{
    int n;
    cin>>n;
    int *a= new int[n];
    int *p= new int [n];
    for(int i= 0; i<n; i++){
        cin>>a[i];
    }


    int max= -200000000;
    int min= a[0];
    for(int i= 1; i<n; i++){
        if(max< a[i]- min){
            max= a[i]- min;
        }
        if(a[i]<min){
            min= a[i];
        }
    }

    
    
    cout<<max;
    return 0;
}
