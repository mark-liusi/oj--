#include<iostream>
using namespace std;
int main()
{
    int number_of_numbers;
    cin>>number_of_numbers;

    double *p= new double[number_of_numbers];
    for(int i= 0; i< number_of_numbers; i++){
        cin>>p[i];
        cout<< p[i]<<" ";
    }
    cout<<endl;
    for(int i= 1; i<number_of_numbers; i++){
        double key= p[i];
        int j= i-1;
        while(j>= 0&& p[j]>key){
            p[j+1]= p[j];
            j--;
        }
        p[j+1]= key;
        for(int k= 0; k<number_of_numbers; k++){
            cout<<p[k]<<" ";
        }
        cout<<endl;
    }
    return 0;
}