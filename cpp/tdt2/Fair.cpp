#include<bits/stdc++.h>
using namespace std;
typedef struct Fair
{
    double first;
    long double number;
}Fair;
void change(vector<Fair> &a){
    for(int i= 0; i<a.size(); i++){
        a[i].number= log2(a[i].first);
    }
}
void judge(vector<Fair> &a){
    long long counta= 0;
    for(int i=  1; i<a.size(); i++){
        if(a[i].number>= a[i-1].number){
            continue;
        }else{
            while(a[i].number< a[i-1].number){
                a[i].number*= 2;
                counta++;
                
            }


        }
    }
    cout<<counta<<endl;
    
}
int main()
{
    int number;
    cin>>number;
    for(int i= 0; i<number; i++){
        int n;
        cin>>n;
        vector<Fair> a(n);
        int order= 0;
        for(int j= 0; j<n; j++){
            cin>>a[j].first;
            a[j].number=0;
            if((a[j].first== 1)&&(j>= 1)){
                if(a[j-1].first!= 1){
                    order= 1;
                    cout<<"-1"<<endl;
                }
            }
        }
        if(order== 0){
            change(a);
            judge(a);
        }
        
    }
    return 0;
}