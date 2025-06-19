#include<iostream>
#include<vector>
using namespace std;
int main()
{
    int number;
    cin>>number;
    vector<int> a(number);
    for(int i= 0; i<number; i++){
        cin>>a[i];
    }
    int count= 0;
    for(int i= 0; i<number; i++){
        for(int j= i+1; j<number; j++){
            if(a[j]< a[i]){
                int temp= a[i];
                a[i]= a[j];
                a[j]= temp;
                count++;
            }
        }
    }
    cout<<count;
    return 0;
}