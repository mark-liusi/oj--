#include<iostream>
#include<vector>
using namespace std;
int Partition(vector<pair<int, string>> &a, int p, int r);
void Quicksort(vector<pair<int, string>> &a, int p, int r);
void merge_sort(vector<pair<int, string>> &a, int left, int right);
void merge(vector<pair<int, string>> &a, int left, int mid, int right);
int main()
{
    int number;
    cin>>number;
    vector<pair<int, string>> a(number);
    vector<pair<int, string>> b(number);
    for(int i= 0; i<number; i++){
        cin>>a[i].second>>a[i].first;
        b[i]= a[i];
    }
    Quicksort(a, 0, number-1);
    merge_sort(b, 0, number);
    int judge= 0;
    for(int i= 0; i<number; i++){
        if(a[i]!= b[i]){
            judge= 1;
            cout<<"Not stable"<<endl;
            break;
        }
    }
    if(judge== 0){
        cout<<"Stable"<<endl;
    }
    //cout<<"Not stable"<<endl;
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
            pair<int, string> temp= a[i];
            a[i]= a[j];
            a[j]= temp;
        }
    }
    pair<int, string> temp= a[i+1];
    a[i+1]= a[r];
    a[r]= temp;
    return i+1;
}
void merge(vector<pair<int, string>> &a, int left, int mid, int right)
{

    int n1= mid- left;
    int n2= right- mid;
    vector<pair<int, string>> L(n1+1);
    vector<pair<int, string>> R(n2+1);
    for(int i= 0; i<n1; i++){
        L[i]= a[left+i];
    }
    for(int i= 0; i<n2; i++){
        R[i]= a[mid+i];
    }
    L[n1].first= 1e9;
    R[n2].first= 1e9;
    int i= 0, j= 0;
    for(int k= left; k<right; k++){
        if(L[i].first<= R[j].first){
            a[k]= L[i];
            i++;
        }else{
            a[k]= R[j];
            j++;
        }
    }
}
void merge_sort(vector<pair<int, string>> &a, int left, int right)
{
    if(left+1< right){
        int mid= (left+ right)/2;
        merge_sort(a, left, mid);
        merge_sort(a, mid, right);
        merge(a, left, mid, right);
    }
}
