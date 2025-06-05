#include<iostream>
#include<vector>
using namespace std;
int Partition(vector<pair<int, string>> &a, int p, int r);
void Quicksort(vector<pair<int, string>> &a, int p, int r);
int main()
{
    int number;
    cin>>number;
    vector<pair<int, string>> a(number);
    vector<pair<int, string>> b(number);
    for(int i= 0; i<number; i++){
        cin>>a[i].second>>a[i].first;
    }
    Quicksort(a, 0, number-1);
    for(pair<int, string> i: a ){
        cout<<i.second<<" "<<i.first<<endl;
    }
    return 0;
}
void Quicksort(vector<pair<int, string>> &a, int p, int r)
{
    if(p<r){
       int q = Partition(a, p, r);
       Quicksort(a, p, q-1);
       Quicksort(a, q+1, r);
    }
}
int Partition(vector<pair<int, string>> &a, int p, int r)
{
    int x= a[r].first;
    int i= p-1;
    for(int j= p; j<=r-1; j++){
        if(a[j].first<= x){
            i++;
            vector<pair<int, string>> temp(1);
            temp[0]= a[i];
            a[i]= a[j];
            a[j]= temp[0];
        }
    }
    vector<pair<int, string>> temp(1);
    temp[0]= a[i+1];
    a[i+1]= a[r];
    a[r]= temp[0];
    return i+1;
}