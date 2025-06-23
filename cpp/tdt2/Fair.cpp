#include<bits/stdc++.h>
using namespace std;
typedef struct Fair
{
    double first;
    int number;
}Fair;
void change(vector<Fair> &a){
    for(int i= 0; i<a.size(); i++){
        a[i].number= log10(a[i].first);
        a[i].first/= pow(10, a[i].number);
    }
}
void judge(vector<Fair> &a){
    int counta= 0;
    for(int i=  1; i<a.size(); i++){
        if((a[i].number>a[i-1].number)){
            continue;
        }else if(a[i].number== a[i-1].number){
            if(a[i].first>= a[i-1].first){
                continue;
            }else{
                if(a[i].number== 0){
                    while(a[i].first<a[i-1].first){
                        a[i].first*= a[i].first;
                        counta++;
                    }
                    if(a[i].first>= 10){
                        a[i].number= log10(a[i].first);
                        a[i].first/= pow(10, a[i].number);
                    }
                }else{
                    a[i].number*= 2;
                    a[i].first*= a[i].first;
                    counta++;
                }
            }
        }else{
            if(a[i].number== 0){
                while (a[i].first<10)
                {
                    a[i].first*= a[i].first;
                    counta++;
                }
                a[i].number= log10(a[i].first);
                a[i].first/= pow(10, a[i].number);
            }
            
            while(a[i].number<a[i-1].number){
                
                a[i].number*= 2;
                a[i].first*= a[i].first;
                counta++;
            }
            if(a[i].number== a[i].number){
                if(a[i].first<a[i-1].first){
                    a[i].number*= 2;
                    a[i].first*= a[i].first;
                    counta++;
                }
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