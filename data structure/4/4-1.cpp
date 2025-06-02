#include<iostream>
using namespace std;
typedef struct sign
{
    int m;
    int n= 0;
}sign;
void binary(int *a, int m, sign b, int *cnt);
int main()
{
    int m, n;
    cin>>m;
    int *a= new int[m];
    for(int i= 0; i< m; i++){
        cin>>a[i];
    }
    cin>>n;
    sign *b= new sign[n];
    for(int i= 0; i< n; i++){
        cin>>b[i].m;
    }

    int *cnt= new int;
    cnt[0]= 0;
    for(int i= 0; i<n; i++){
        binary(a, m, b[i], cnt);
    }
    cout<<*cnt;
    return 0;
}
void binary(int *a, int m, sign b, int *cnt)
{
    int left = 0, right = m - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (a[mid] == b.m) {
            cnt[0]++;
            return;
        } else if (a[mid] < b.m) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return; 
}