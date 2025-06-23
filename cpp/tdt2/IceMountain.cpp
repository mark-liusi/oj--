#include<bits/stdc++.h>
using namespace std;
int change(string tail){
    int sum= 0;
    for(int i= 0; i<tail.size(); i++){
        sum+= (tail[i]-'0')* pow(2, tail.size()-1-i);
    }
    return sum;
}
int main()
{
    string code;
    cin>>code;
    string content= "";
    for(int j= 0; j<code.size(); j++){
        if(code[j]!= '0'&& code[j]!= '1'){
            cout<<"Error";
            return 0;
        }
    }
    int i= 0;
    while (code[i]!= '\0')
    {
        string head= code.substr(i, 3);
        if(head== "101"){
            i+= 3;
            string tail= "";
            if(i+5> code.size()){
                cout<<"Error";
                return 0;
            }
            tail= code.substr(i, 5);
            i+= 5;
            if(change(tail)>=26){
                cout<<"Error";
                return 0;
            }
            content+= static_cast<char>('A'+ change(tail));
        }else if(head== "111"){
            i+= 3;
            if(i+5> code.size()){
                cout<<"Error";
                return 0;
            }
            i+= 5;
            content+= " ";
        }else if(head[0]== '0'){
            i+= 1;
            if(i+15> code.size()){
                cout<<"Error";
                return 0;
            }
            int number1= change(code.substr(i, 7))/2; i+= 7;
            if(code[i]!= '0'){
                cout<<"Error";
                return 0;
            }
            int number2= change(code.substr(i+1, 7))/2; i+= 8;
            content+= to_string(number1+number2);
        }else{
            cout<<"Error";
            return 0;
        }
    }
    cout<<content;
    return 0;
}
