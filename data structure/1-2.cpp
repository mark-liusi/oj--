#include<iostream>
using namespace std;
int main()
{
    int numberone, numbertwo;
    cin>>numberone;
    cin>>numbertwo;
    if(numberone< numbertwo){
        int numberthree= numberone;
        numberone= numbertwo;
        numbertwo= numberthree;
    }
    while (numberone%numbertwo!= 0)
    {
        int r= numberone% numbertwo;
        numberone= numbertwo;
        numbertwo= r;
    }
    cout<<numbertwo;
    return 0;
    
}
