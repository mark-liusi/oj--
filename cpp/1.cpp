#include <iostream>
#include <vector>
using namespace std;

bool fuhao(char c) {
    string ops = "+-*/><=(),;{}()";
    return ops.find(c) != string::npos;
}

int main() {
    string input;
    getline(cin, input);
    vector<string> a;
    vector<string> b;
    int i = 0;

    cout<<"Token :";
    while (i < input.size()) {
        if (isalpha(input[i])) {
            string token;
            while (i < input.size() && isalnum(input[i])) {
                token += input[i];
                i++;
            }
            if(token== "int"||token== "void"|| token== "break"|| token== "float"||token== "while"|| token=="do"||token== "struct"||token== "const"||token== "case"||token== "for"||token== "return"||token== "if"||token== "default"||token== "else"){
                if(token== "int"){
                    cout<<"(K 1)";
                }else if(token== "void"){
                    cout<<"(K 2)";
                }else if(token== "break"){
                    cout<<"(K 3)";
                }else if(token== "float"){
                    cout<<"(K 4)";
                }else if(token== "while"){
                    cout<<"(K 5)";
                }else if(token== "do"){
                    cout<<"(K 6)";
                }else if(token== "struct"){
                    cout<<"(K 7)";
                }else if(token== "const"){
                    cout<<"(K 8)";
                }else if(token== "case"){
                    cout<<"(K 9)";
                }else if(token== "for"){
                    cout<<"(K 10)";
                }else if(token== "return"){
                    cout<<"(K 11)";
                }else if(token== "if"){
                    cout<<"(K 12)";
                }else if(token== "default"){
                    cout<<"(K 13)";
                }else if(token== "else"){
                    cout<<"(K 14)";
                }
                continue;
            }
            int judge= 0;
            
            for(int e=0; e<a.size(); e++){
                if(token== a[e]){
                    cout << "(I " << e+1 << ")";
                    judge= 1;
                    break;
                }
            }
            if(judge== 0){
                a.push_back(token);
                cout << "(I " << a.size() << ")";
            }
            
        }
        else if (isdigit(input[i])) {
            string num;
            while (i < input.size() && isdigit(input[i])) {
                num += input[i];
                i++;
            }
            int judge= 0;
            
            for(int e=0; e<b.size(); e++){
                if(num== b[e]){
                    cout << "(C " << e+1 << ")";
                    judge= 1;
                    break;
                }
            }
            if(judge== 0){
                b.push_back(num);
                cout << "(C " << b.size() << ")";
            }
        }
        else if (fuhao(input[i])) {
            if (i + 1 < input.size()) {
                string current= "";
                current += input[i];
                current += input[i+1];
                if (current == "++" ) {
                    cout<<"(P 14)";
                    i += 2;
                    continue;
                }else if (current == "==" ) {
                    cout<<"(P 5)";
                    i += 2;
                    continue;
                }if (current == "<=" ) {
                    cout<<"(P 6)";
                    i += 2;
                    continue;
                }
                    
                
            }
            switch(input[i]) {
                case '+': cout << "(P 8)"; break;
                case '-': cout << "(P 1)"; break;
                case '*': cout << "(P 9)"; break;
                case '/': cout << "(P 2)"; break;
                case '(': cout << "(P 3)"; break;
                case ')': cout << "(P 4)"; break;
                case '<': cout << "(P 7)"; break;
                case '>': cout << "(P 10)"; break;
                case '=': cout << "(P 11)"; break;
                case ',': cout << "(P 12)"; break;
                case ';': cout << "(P 13)"; break;
                case '{': cout << "(P 15)"; break;
                case '}': cout << "(P 16)"; break;
            }
            i++;
        }
        else {
            i++;  
        }
    }

    cout << "\nI :";
    for (string s1 : a){
        cout << s1 << " ";
    } 
    cout << "\nC :";
    for (string s2 : b) {
        cout << s2 << " ";
    }
    
    return 0;
}