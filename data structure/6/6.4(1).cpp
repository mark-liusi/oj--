#include<iostream>
#include<vector>
using namespace std;
int main()
{
    int number;
    cin>>number;
    vector<pair<int, int>> a(number);
    vector<pair<int, int>> b(number);
    for(int i= 0; i<number; i++){
        cin>>a[i].first;
        a[i].second= 0;
        b[i]= a[i];
    }
    for(int i= 0; i<number; i++){
        a[i].second= i;
    }
    int sum= 0;
    for(int i= 0; i<number; i++){
        int min= a[i].first;
        int position= i;
        for(int j= i+1; j<number; j++){
            if(a[j].first< min){
                min= a[j].first;
                position= j;
            }
        }
        // if(position!= i){
        //     sum+= min+ a[i];
        // }
        a[position]= a[i];
        a[i].first= min;
    }
    for(int i= 0; i<number; i++){
        for(int j= 0; j<number; j++){
            if(b[j].first== a[i].first){
                b[j].second= a[i].second;
                break;
            }
        }
    }

    
}