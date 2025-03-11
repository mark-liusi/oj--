#include<iostream>
using namespace std;
void insertionsort(int *a, int n, int g, int *cnt);
void shellsort(int *a, int n, int *cnt);
int main()
{
    int n;
    cin>>n;
    int *a= new int[n];
    for(int i= 0; i<n; i++){
        cin>>a[i];
    }
    int *cnt= new int[1];
    cnt[0]= 0;
    shellsort(a, n, cnt);
    cout<<cnt[0]<<endl;
    for(int i= 0; i<n; i++){
        cout<<a[i]<<endl;
    }
    return 0;

}
void insertionsort(int *a, int n, int g, int *cnt)
{
    for(int i= g; i<n; i++){
        int v= a[i];
        int j= i- g;
        while (j>= 0&& a[j]>v)
        {
            a[j+g]= a[j];
            j= j-g;
            cnt[0]++;
        }
        a[j+g]= v;
    }
}
void shellsort(int *a, int n, int *cnt)
{   
    int count= 1;
    int m= 1;
    while(3*count+1< n){
        count= 3*count+1;
        m++;
    }
    cout<<m<<endl;

    int *p= new int[m];
    int j= 0;
    p[0]= 1;
    for(int i= 1 ; i<m; i++){
        p[i]= 3*p[i-1]+1;
    }
    int *q= new int[m];
    for(int j= 0; j<m; j++){
        q[j]= p[m-1-j];
        cout<<q[j]<<" ";
    }

    cout<<endl;
    for(int i= 0; i<m; i++){
        insertionsort(a, n, q[i], cnt);
    }
}