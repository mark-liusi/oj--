#include<iostream>
using namespace std;
int main()
{
    int n;
    cin>>n;
    int *a= new int[n];
    for(int i= 0; i<n; i++){
        cin>>a[i];
    }

    int number= 0;
    for(int i= 0; i<n; i++){
        for(int j= n-1; j>i; j--){
            if(a[j]< a[j-1]){
                int temp= a[j];
                a[j]= a[j-1];
                a[j-1]= temp;
                number++;
            }
        }
    }

    for(int i= 0; i<n; i++){
        cout<<a[i]<<" ";
    }
    cout<<endl<<number;
    return 0;
}