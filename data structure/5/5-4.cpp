#include<iostream>
#include<vector>
using namespace std;
void merge(vector<int> &a, int &cnt, int left, int right);
void merge_sort(vector<int> &a, int left, int right, int &cnt);
int main()
{
    int n;
    cin>>n;
    vector<int> a(n);
    for(int i=0; i<n; i++){
        cin>>a[i];
    }
    int cnt= 0;

    merge_sort(a, 0, n, cnt);
    // for(int i: a){
    //     cout<<i<<" ";
    // }
    cout<<cnt;
    return 0;
}
void merge(vector<int> &a, int &cnt, int left, int right)
{
    int mid= (left+ right)/2;
    int n1= mid- left;
    int n2= right- mid;
    vector<int> L(n1+1);
    vector<int> R(n2+1);
    L[n1]= 1e9;
    R[n2]= 1e9;
    for(int i= 0; i<n1; i++){
        L[i]= a[left+ i];
    }
    for(int i= 0; i<n2; i++){
        R[i]= a[mid+ i];
    }

    int i= 0, j= 0;
    for(int k = left; k < right; k++) {
        if (L[i] <= R[j]) {
            a[k] = L[i++];
        } else {
            a[k] = R[j++];
            cnt += mid- i+ 1;
        }
    }
}
void merge_sort(vector<int> &a, int left, int right, int &cnt)
{
    if(left+1< right){
        int mid= (left+ right)/2;
        merge_sort(a, left, mid, cnt);
        merge_sort(a, mid, right, cnt);
        merge(a, cnt, left, right);
    }
}