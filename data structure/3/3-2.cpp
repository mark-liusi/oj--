#include<iostream>
using namespace std;
typedef struct todo{
    string name;
    int time;
    todo *next;
}todo;
int main()
{
    int n, time;
    cin>>n;
    cin>>time;
    int number= n;
    todo *head= new todo;
    todo *p1= new todo;
    cin>>p1->name;
    cin>>p1->time;
    head->next= p1;
    todo *p2= p1;
    for(int i= 0; i<n-1; i++){
        todo *p1= new todo;
        p2->next= p1;
        cin>>p1->name;
        cin>>p1->time;
        p2= p1;
    }
    p2->next= NULL;

    p1= head->next;
    //cout<<head->next->name<<endl;
    // while(p1!= NULL){
    //     cout<<p1->name<<endl;
    //     p1= p1->next;
    // }
    int sumtime= 0;
    todo *end= p2;
    while(n>0){
        if(p1->time> time){
            if(p1== end){
                cout<<p1->name<<" "<<sumtime+p1->time<<endl;
                break;
            }
            sumtime+= time;
            p1->time-= time;
            head->next= p1->next;
            end->next= p1;
            end= p1;
            p1= head->next;
        }else if(p1->time<= time){
            sumtime+= p1->time;
            cout<<p1->name<<" "<<sumtime<<endl;
            head->next= p1->next;
            p1= head->next;
            n--;
        }
    }
    return 0;
}
//指针变量的赋值