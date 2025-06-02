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
        int mini= i;
        for(int j= i; j<n; j++){
            if(a[j]<a[mini]){
                mini= j;
            }
        }
        if(mini!= i){
            int temp= a[i];
            a[i]= a[mini];
            a[mini]= temp;
            number++;
        }
    }

    for(int i= 0; i<n; i++){
        cout<<a[i]<<" ";
    }
    cout<<endl<<number;
    return 0;
}