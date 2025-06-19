#include<iostream>
#include<vector>
using namespace std;
int main()
{
    int number;
    cin>>number;
    int sum= 0;
    vector<int> a(number);
    for(int i= 0; i<number; i++){
        cin>>a[i];
        sum+= a[i];
    }
    
    int ave= sum/number;
    int i= 0;
    int count= 0;
    while(i<number-1){
        if(a[i]== ave){
            i++;
        }else{
            int temp= a[i]-ave;
            a[i+1]+= temp;
            count++;
            i++;
        }
        
    }
    
    cout<<count;
}