#include<iostream>
#include<string>
using namespace std;
typedef struct area
{
    int size;
    int left;
    int right;
    area *next;
}area;

void resort(area *head);
int main()
{
    string a= "";
    cin>>a;

    int position= 0;
    int di[20001]= {0};
    area *head= new area;
    area *p2= head;
    int sum= 0;
    int number= 0;

    for(int i= 0; i<a.size(); i++){
        if(a[i]== '\\'){
            di[position]= i;
            position++;
        }else if(a[i]== '_'){
            continue;
        }else if(a[i]== '/'){
            if(position== 0){
                continue;
            }
            area *p1= new area;
            position--;
            p1->size= i- di[position];
            p1->left= di[position];
            p1->right= i;
            sum+= p1->size;
            p2->next= p1;
            p2= p1;
        }
    }
    p2->next= NULL;

    
    
    cout<<sum<<endl;
    area *current= head->next;
    resort(current);

    current= head->next;
    int n= 0;
    while (current!= NULL)
    {
        if(current->size!= 0){
            n++;
        }   
        current= current->next;
    }
    cout<<n<<" ";
    
    current= head->next;

    while (current!= NULL)
    {
        if(current->size!= 0){
            cout<<current->size<<" ";
        }   
        current= current->next;
    }
    
    
    return 0;
}
void resort(area *head)
{
    area *current= head;
    if(current== NULL){
        return;
    }
    while(current->next!= NULL){
        if(current->left> current->next->left && current->right< current->next->right){
            current->next->size+= current->size;
            current->size= 0;
        }
        current= current->next;
    }
    current= head;
    int n= 0;
    while(current->next!= NULL){
        if (current->next->size== 0)
        {
            if(current->next->next== NULL){
                current->next= NULL;
            }
            current->next= current->next->next;
            n++;
        }
        current= current->next;
    }
    if(n== 0){
        return;
    }else{
        resort(head);
    }
}