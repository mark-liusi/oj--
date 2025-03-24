#include <iostream>
#include<string>
using namespace std;
void find(string str, string add);
void tonext(string add, int *next);
int main() {
    int n;
    cin>>n;
    string dic = "";
    for(int i= 0; i<n; i++){
        string order;
        cin>>order;
        string add;
        cin>>add;
        if(order== "insert"){
            dic+= add;
            dic+= " ";
        }else if(order== "find"){
            find(dic, add);
        }
    }
    return 0;
}
// void tonext(string add, int *next)
// {
//     int i= 0;  int j= 1;
//     next[0]= 0;
//     while(j<= add.size()){
//         if(add[j]== add[i]){
//             next[j]= i+1;
//             i++, j++;
//         }else{
//             if(i== 0){
//                 next[j]= 0;
//                 j++;
//             }else{
//                 i= next[i-1];
//             }
//         }
        
//     }
// }
// void find(string str, string add)
// {
//     int *next= new int[add.size()];
//     tonext(add, next);
//     int i= 0, j= 0;
//     while(j< str.size()){
//         if(add[i]== str[j]){
//             i++, j++;
//         }else{
//             if(i== 0){
//                 j++;
//             }else{
//                 i= next[i-1];
//             }
//         }
//         if(i== add.size()){
//             cout<<"yes"<<endl;
//             break;
//         }else if(j== str.size()){
//             cout<<"no"<<endl;
//             break;
//         }else{
//             continue;
//         }
//     }
// }
void find(string str, string add)
{
    int i= 0, j= 0;
    while (j<= str.size())
    {
        j= str.find(' ', i+1);
        string test= str.substr(i, j-i);
        if(test== add){
            cout<<"yes"<<endl;
            return ;
        }else{
            i= j+1;
        }
    }
    cout<<"no"<<endl;
}   