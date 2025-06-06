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
    int sum= 0;
    for(int i= 0; i<number; i++){
        int min= a[i];
        int position= i;
        for(int j= i+1; j<number; j++){
            if(a[j]< min){
                min= a[j];
                position= j;
            }
        }
        if(position!= i){
            sum+= min+ a[i];
        }
        a[position]= a[i];
        a[i]= min;
    }
    cout<<sum;
    return 0;
}