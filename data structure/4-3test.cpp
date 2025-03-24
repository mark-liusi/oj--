#include<iostream>
#include<string>
using namespace std;
void tonext(string add, int *next)
{
    int i= 0;  int j= 1;
    next[0]= 0;
    while(j<= add.size()){
        if(add[j]== add[i]){
            next[j]= i+1;
            i++, j++;
        }else{
            if(i== 0){
                next[j]= 0;
                j++;
            }else{
                i= next[i-1];
            }
        }
        
    }
}
int main()
{
    string a= "abcaby";
    int *next= new int[6];
    tonext(a, next);
    for(int i= 0; i<6; i++){
        cout<<next[i]<<" ";
    }
    return 0;
}