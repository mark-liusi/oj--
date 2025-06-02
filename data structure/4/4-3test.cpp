#include<iostream> 
#include<vector> 
#include<set>
using namespace std;
int main()
{
    int n;
    cin>>n;
    set<string> a;
    while(n> 0){
        n--;
        string order;
        string str;
        cin>>order>>str;
        if(order== "insert"){
            a.insert(str);
        }else if(order== "find"){
            auto judge= a.find(str);
            if(judge== a.end()){
                cout<<"no"<<endl;
            }else{
                cout<<"yes"<<endl;
            }
        }
        
    }
    return 0;
}