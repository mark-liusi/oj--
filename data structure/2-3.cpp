#include<iostream>
#include<string>
#include<string.h>
#include<cstdlib>
using namespace std;
typedef struct item
{
    int number;
    string word;
    int order;
}item;
void output(item *p, int n);
void judgement(item *a, item *c, int n);
void order(item *a, int n);
int main()
{
    int n;
    cin>>n;
    item *a= new item[n];
    item *b= new item[n];
    item *c= new item[n];
    item *d= new item[n];
    
    for(int i= 0; i<n; i++){
        string input;
        cin>>input;
        a[i].word= input[0];
        a[i].number= stoi(input.substr(1, 1));
        a[i].order= 0;
        b[i]= a[i];
        c[i]= a[i];
        d[i]= a[i];
    }

    for(int i= 0; i<n; i++){
        for(int j= n-1; j>i; j--){
            if(a[j].number<a[j-1].number){
                item temp= a[j];
                a[j]= a[j-1];
                a[j-1]= temp;
            }
        }
    }
    order(a, n);
    for(int i=0; i<n; i++){
        int mini= i;
        for(int j= i; j<n; j++){
            if(b[j].number<b[mini].number){
                mini= j;
            }
        }
        if(mini!= i){
            item temp= b[i];
            b[i]= b[mini];
            b[mini]= temp;
        }
    }
    order(b, n);
    order(c, n);
    order(d,n);
    
    output(a, n);
    
    judgement(a, c, n);

    output(b, n);
    judgement(b, d, n);
    

    return 0;
}
void output(item *p, int n){
    for(int i= 0; i<n; i++){
        cout<<p[i].word<<p[i].number<<" ";
    }
}
void judgement(item *a, item *c, int n){
    int judge= 0;
    for(int i= 0; i<n; i++){
        if(c[i].order== 0){
            continue;
        }else{
            for(int j= 0; j<n; j++){
                if(a[j].order!= 0&& (c[i].number== a[j].number)&&c[i].order!= 0){
                    if(c[i].word!= a[j].word){
                        judge++;
                        break;
                    }else{
                        c[i].order= 0;
                        a[j].order= 0;
                    }
                }else{
                    continue;
                }
            }
        }
        if(judge!= 0){
            break;
        }
    }
    if(judge==0){
        cout<<endl<<"Stable"<<endl;
    }else{
        cout<<endl<<"Not stable"<<endl;
    }
}
void order(item *a, int n){
    for(int i= 0; i< n; i++){
        if(a[i].order!= 0){
            continue;
        }else{
            int order= 2;
            for(int j= i+1; j< n; j++){
                if(a[j].number== a[i].number){
                    a[i].order=1;
                    a[j].order= order;
                    order++;
                }
            }
        }
        
    }
}