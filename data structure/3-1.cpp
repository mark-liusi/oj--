#include<iostream>
#include<string>
using namespace std;
int main()
{
    string s= "";
    string all= "";
    int p[100]= {0};
    int m= 0;
    while(cin>>s){
        if(s== "+"||s== "-"||s== "*"){
            m--;
            if(s== "+"){
                p[m-1]= p[m]+p[m-1];
                p[m]= 0;
            }else if(s== "-"){
                p[m-1]= p[m-1]-p[m];
                p[m]= 0;
            }else if(s== "*"){
                p[m-1]= p[m]*p[m-1];
                p[m]= 0;
            }
        }
        else if(stoi(s)>=-1e9 && stoi(s)<=1e9){
            p[m]= stoi(s);
            m++;
        }else{
            continue;
        }
    }
    int sum= 0;
    for(int i= 0; i<m; i++){
        sum+= p[i];
    }
    cout<<sum;
    return 0;
}