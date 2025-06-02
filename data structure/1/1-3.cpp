#include<iostream>
#include<math.h>
using namespace std;
int main()
{
    int N;
    cin>>N;
    int *a= new int[N];
    for (size_t i = 0; i < N; i++)
    {
        cin>>a[i];
    }
    int number= 0;
    for(int i= 0; i<N; i++){
        for(int j= 2; j<= pow(double(a[i]), 0.5); j++){
            if (a[i]%j== 0)
            {   
                //cout<<a[i];
                number++;
                break;
            }
            
        }
    }
    cout<<N-number;
    return 0;
}