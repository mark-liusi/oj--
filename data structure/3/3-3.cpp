#include<iostream>
using namespace std;
typedef struct number{
    int x;
    number *next;
}number;
void insert(number *head, int num);
void deleted(number *head, int num);
void deletefirst(number *head);
void deletelast(number *head);
int main()
{
    int n;
    cin>>n;
    number *head= new number;
    head->next= NULL;
    number *p1= head;
    for(int i= 0; i<n; i++){
        string order;
        cin>>order;
        if(order== "insert"){
            int num;  
            cin>>num;
            insert (head, num);
        }else if(order== "delete"){
            int num;
            cin>>num;
            deleted(head, num);
        } 
        else if(order== "deleteFirst") deletefirst(head);
        else if(order== "deleteLast") deletelast(head);
    }
    head= head->next;
    while(head!= NULL){
        cout<<head->x<<" ";
        head= head->next;
    }
    return 0;
}
void insert(number *head, int num)
{
    number *p1= new number;
    p1->x= num;
    p1->next= head->next;
    head->next= p1;
}
void deleted(number *head, int num)
{
    number *p1= head->next;
    number *p2= head;
    while(p1!= NULL){
        if(p1->x== num){
            p2->next= p1->next;
            break;
        }else{
            p2= p1;
            p1= p1->next;
        }
        
    }
}
void deletefirst(number *head)
{
    head->next= head->next->next;
}
void deletelast(number *head)
{
    if (head->next == NULL) return;
    number *p1 = head;
    while (p1->next->next != NULL) {
        p1 = p1->next;
    }
    delete p1->next;
    p1->next = NULL;
    // if (head->next == NULL) return;
    // number *p1= head->next;
    // while(1){
    //     if(p1->next->next== NULL){
    //         p1->next= NULL;
    //         break;
    //     }
    //     p1= p1->next;
    // }
}
/*4
insert 1
insert 2
insert 3
delete 3
deleteLast
insert 4*/