#include<iostream>
#include<set>
#include<vector>
using namespace std;
int main()
{
    int number_of_numbers;
    cin>>number_of_numbers;
    vector<int> a(number_of_numbers);
    for(int i=0; i<number_of_numbers; i++){
        cin>>a[i];
    }

    set<int> c;
    for(int i= 0; i<(1<<number_of_numbers); i++){
        int sum= 0;
        for(int j= 0; j<number_of_numbers; j++){
            if(i& (1<<j)){
                sum+= a[j];
            }
        }
        c.insert(sum);
    }

    int number_of_sum;
    cin>>number_of_sum;
    for(int i=0; i<number_of_sum; i++){
        int number;
        cin>>number;
        if(c.find(number)!= c.end()){
            cout<<"yes"<<endl;
        }else{
            cout<<"no"<<endl;
        }
    }
    return 0;
}