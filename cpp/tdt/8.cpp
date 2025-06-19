#include<iostream>
#include<vector>
#include<algorithm>
#include<queue>
using namespace std;
int main()
{
    int number, a, b;
    cin>>number>>a>>b;
    priority_queue<int> c;
    for(int i= 0; i<number; i++){
        int temp;
        cin>>temp;
        c.push(temp);
    }
    int count= 0;
    while(!c.empty()){
        priority_queue<int> temp;
        if(c.top()> a+b){
            temp.push(c.top()-a-b);
        }
        c.pop();
        while (!c.empty())
        {
            if(c.top()>a){
                temp.push(c.top()-a);
            }
            c.pop();
        }
        count++;
        c= temp;
    }
    
    cout<<count;
    // for(int i:c){
    //     cout<<i<<endl;
    // }
    return 0;
}