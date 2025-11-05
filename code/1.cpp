#include<iostream>
using namespace std;
int main()
{
    
    for i = 1 to A.length-1
    key = A[i]
    /* insert A[i] into the sorted sequence A[0,...,j-1] */
    j = i - 1
    while j >= 0 and A[j] > key
        A[j+1] = A[j]
        j--
    A[j+1] = key

}