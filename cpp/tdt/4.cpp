#include<iostream>
#include<vector>
#include<set>
#include<cmath>
using namespace std;
int main()
{
    int number1;
    int number2;
    cin>>number1>>number2;
    vector<int> a(number1);
    for(int i= 0; i<number1; i++){
        cin>>a[i];
    }

    vector<int> b;
    for(int i= 0; i<pow(2, number1); i++){
        int position= 0;
        int temp= i;
        while(temp!= 0){
            if((1&temp)== 1){
                position++;
            }
            temp= temp>>1;
        }
        if(position== number2){
            b.push_back(i);
        }
    }
    vector<int> c;
    for(int i: b){
        int temp= i;
        int current= 0;
        int sum= 0;
        while(temp!= 0){
            if((1&temp)==1){
                sum+= a[current];
            }
            current++;
            temp= temp>>1;
        }
        c.push_back(sum);
        
    }
    int count= 0;
    for(int i :c){
        int judge= 0;
        for(int j= 2; j<sqrt(static_cast<double>(i)); j++){
            if(i%j== 0){
                judge= 1;
                break;
            }
        }
        if(judge== 0){
            count++;
        }
    }
    cout<<count;
    return 0;
}