#include<iostream>
#include<vector>
using namespace std;
void merge(vector<int> &a, int left, int mid, int right,int& b);
void merge_sort(vector<int> &a, int left, int right, int &b);
int main()
{
    int number;
    cin>>number;
    vector<int> a(number);
    for(int i= 0; i<number; i++){
        cin>> a[i];
    }
    int count= 0;
    merge_sort(a, 0, number, count);
    for(int i= 0; i<number; i++){
        cout<< a[i]<<" ";
    }
    cout<<endl<<count;
    
    return 0;
}
/*Merge(A, left, mid, right)
  n1 = mid - left;
  n2 = right - mid;
  create array L[0...n1], R[0...n2]
  for i = 0 to n1-1
    do L[i] = A[left + i]
  for i = 0 to n2-1
    do R[i] = A[mid + i]
  L[n1] = SENTINEL
  R[n2] = SENTINEL
  i = 0;
  j = 0;
  for k = left to right-1
    if L[i] <= R[j]
      then A[k] = L[i]
           i = i + 1
      else A[k] = R[j]
           j = j + 1
*/
void merge(vector<int> &a, int left, int mid, int right, int& b)
{

    int n1= mid- left;
    int n2= right- mid;
    vector<int> L(n1+1);
    vector<int> R(n2+1);
    for(int i= 0; i<n1; i++){
        L[i]= a[left+i];
    }
    for(int i= 0; i<n2; i++){
        R[i]= a[mid+i];
    }
    L[n1]= 1e9;
    R[n2]= 1e9;
    int i= 0, j= 0;
    for(int k= left; k<right; k++){
        if(L[i]<= R[j]){
            b++;
            a[k]= L[i];
            i++;
        }else{
            b++;
            a[k]= R[j];
            j++;
        }
    }
}
/*Merge-Sort(A, left, right){
  if left+1 < right
    then mid = (left + right)/2;
         call Merge-Sort(A, left, mid)
         call Merge-Sort(A, mid, right)
         call Merge(A, left, mid, right)
*/
void merge_sort(vector<int> &a, int left, int right, int &b)
{
    if(left+1< right){
        int mid= (left+ right)/2;
        merge_sort(a, left, mid, b);
        merge_sort(a, mid, right, b);
        merge(a, left, mid, right, b);
    }
}