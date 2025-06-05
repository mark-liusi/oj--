#include<iostream>
#include<vector>
using namespace std;
int Partition(vector<int> &a, int p, int r);
int main()
{
    int number;
    cin>>number;
    vector<int> a(number);
    for(int i= 0; i<number; i++){
        cin>>a[i];
    }
    int j= Partition(a, 0, number-1);
    for(int i= 0; i<j; i++){
        cout<<a[i]<<" ";
    }
    cout<<"["+to_string(a[j])+"] ";
    for(int i= j+1; i<number; i++){
        cout<<a[i]<<" ";
    }
    // for(int i:a){
    //     if(i== a[j]){
    //         cout<<"["+to_string(i)+"] ";
    //     }else{
    //         cout<<i<<" ";
    //     }
        
    // }
    return 0;

}
int Partition(vector<int> &a, int p, int r)
{
    int x= a[r];
    int i= p-1;
    for(int j= p; j<=r-1; j++){
        if(a[j]<= x){
            i++;
            int temp= a[i];
            a[i]= a[j];
            a[j]= temp;
        }
    }
    int temp= a[i+1];
    a[i+1]= a[r];
    a[r]= temp;
    return i+1;
}
/*i = p-1
for j = p to r-1
    do if A[j] <= x
       then i = i+1
           exchange A[i] and A[j] 
exchange A[i+1] and A[r]
return i+1
12
9 5 8 7 4 2 6 11 21 13 19 12*/