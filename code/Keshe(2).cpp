#include<iostream>
#include<vector>
#include<map>

using namespace std;

enum ERROR{NOERROR, ERROR0};
enum cat { f, c, t, d,v,vn,vf };
//f(函数)，c(常量)，t(类型)，d(域名)，v, vn, vf(变量，换名形参，赋值形参)；

struct procedure {

};
struct struct_ {

};
struct array_ {

};
struct type_ {

};
struct symbol {
	string NAME;
	int TYP;
	cat CAT;
	//ADDR

};
//活动记录
// ？

//长度表
vector<int> LENFL;
//函数表
vector<procedure> PFINFL;
//结构表
vector<struct_> RINFL;
//数组表
vector<array_> AINFL;
//类型表
vector<type_> TYPEL = {};
//关键字表
vector<string> KT = {"program","integer",
"real","char","array","record","end","function",
"var","while","begin","if","then","do","return","not","and","or", "none"};
//界符表
vector<string> PT = {"=","+","-","*","/","<","<=",">",">=","==",":",";","(",")",".",",","[","]"};
//常数表
vector<double> CT ;
//字符串表
vector<string> ST ;
//字符表
vector<char> cT ;
//符号表（标识符表）
vector<symbol> SYNBL;

vector<pair<string, int>> Token;
//string Token= "";
bool fuhao(char c) {
    string ops = "+-*/><=(),;{}().:[]";
    return ops.find(c) != string::npos;
}
typedef struct error_position
{
    int line;
    string word;
    ERROR error;
}error_position;

error_position error_p= {-1, "-1", NOERROR};


int main() {
    string input= "";
    string words;
    while(getline(cin, words)){//将输入的内容读取到words中
        if(words!= "end."){
            input+= words+"\n";
        }else{
            input+= "end.";
            break;
        }
    }
    map<int, string> error;
    error[1]= "非法字符错误";
    error[2]= "标识符格式错误";
    error[3]= "数字格式错误";
    error[4]= "界符错误";
    // vector<pair<string, vector<int>>> SYNBL;
    // vector<pair<double, vector<int>>> CT;
    // vector<pair<string, vector<int>>> KT;//关键字表
    // vector<pair<int, vector<int>>> PT;
    // vector<pair<char, vector<int>>> cT;
    // vector<pair<string, vector<int>>> ST;

    vector<string> iT;//标识符表
    vector<int> line;
    int Line= 1;
    //line.push_back(Line);
    int i = 0;
    //cout<<input[166];
    //cout<<"Token :";
    //i= 166;
    while (i < input.size()) {
        if (isalpha(input[i])) {//如果是字母的话
            string token;
            //首先对SYNBL标识符，KT关键字进行识别，此时字母后面跟字母（关键字或标识符）或者数字（标识符），
            while (i < input.size() && isalnum(input[i])) {
                token += input[i];
                i++;
            }
            //对于关键字表KT进行扫描识别
            if(token== "program"||token== "integer"|| token== "real"|| token== "char"||token== "array"|| token=="record"||token== "end"||token== "function"||token== "case"||token== "var"||token== "while"||token== "begin"||token== "if"||token== "then"||token== "do"||token== "return"||token== "not"||token== "and"||token== "or"){
                line.push_back(Line);
                if(token== "program"){
                    Token.push_back({"KT", {0}});
                }else if(token== "integer"){
                    Token.push_back({"KT", {1}});
                }else if(token== "real"){
                    Token.push_back({"KT", {2}});
                }else if(token== "char"){
                    Token.push_back({"KT", {3}});
                }else if(token== "array"){
                    Token.push_back({"KT", {4}});
                }else if(token== "record"){
                    Token.push_back({"KT", {5}});
                }else if(token== "end"){
                    Token.push_back({"KT", {6}});
                }else if(token== "function"){
                    Token.push_back({"KT", {7}});
                }else if(token== "var"){
                    Token.push_back({"KT", {8}});
                }else if(token== "while"){
                    Token.push_back({"KT", {9}});
                }else if(token== "begin"){
                    Token.push_back({"KT", {10}});
                }else if(token== "if"){
                    Token.push_back({"KT", {11}});
                }else if(token== "then"){
                    Token.push_back({"KT", {12}});
                }else if(token== "do"){
                    Token.push_back({"KT", {13}});
                }else if(token== "return"){
                    Token.push_back({"KT", {14}});
                }else if(token== "not"){
                    Token.push_back({"KT", {15}});
                }else if(token== "and"){
                    Token.push_back({"KT", {16}});
                }else if(token== "or"){
                    Token.push_back({"KT", {17}});
                }else if(token== "none"){
                    Token.push_back({"KT", {18}});
                }
                continue;
            }
            int judge= 0;
            //关键字符已检测完，那么接下来的就是标识符表了，首先进行检测防止标识符存储重复。
            for(int e=0; e<SYNBL.size(); e++){
                if(token== SYNBL[e].NAME){
                    line.push_back(Line);
                    Token.push_back({"SYNBL", {e}});
                    //Token+= "(SYNBL "+ to_string(e+1)+")";
                    //cout << "(SYNBL "<< e+1 << ")";
                    judge= 1;
                    break;
                }
            }
            if(judge== 0){//若没有重复，那么录入新的标识符，并记上新的序号
                SYNBL.push_back({token});
                line.push_back(Line);
                Token.push_back({"SYNBL", {static_cast<int>(SYNBL.size()-1)}});
                //string new_synbl= "(SYNBL " +to_string(SYNBL.size())+ ")";
                //Token+= new_synbl;
                //cout << new_synbl;
            }
            
        }
        else if(input[i]== '\"'){
            string s;//字符串
            while (i < input.size() && input[i]!= '\"') {//防止越界以及多位常数
                s+= input[i];
                i++;
            }
            if (i < input.size()) i++;
            int judge= 0;
            
            for(int e=0; e<ST.size(); e++){//检测是否有重复
                if(s== ST[e]){
                    Token.push_back({"ST", {e}});
                    line.push_back(Line);
                    //Token+= "(ST "+to_string(e+1)+")";
                    //cout << "(ST " << e+1 << ")";
                    judge= 1;
                    break;
                }
            }
            if(judge== 0){
                ST.push_back(s);
                line.push_back(Line);
                Token.push_back({"ST", {static_cast<int>(ST.size()-1)}});
                //Token+= "(ST "+to_string(ST.size())+")";
                // cout << "(ST " << ST.size() << ")";
            }
        }
        else if(input[i]== '\''){
            char s= input[i+1];//字符
            i+= 3;
            int judge= 0;
            
            for(int e=0; e<cT.size(); e++){//检测是否有重复
                if(s== cT[e]){
                    Token.push_back({"cT", {e}});
                    line.push_back(Line);
                    //Token+= "(cT "+to_string(e+1)+")";
                    //cout << "(cT " << e+1 << ")";
                    judge= 1;
                    break;
                }
            }
            if(judge== 0){
                cT.push_back(s);
                line.push_back(Line);
                Token.push_back({"cT", {static_cast<int>(cT.size()-1)}});
                //Token+= "(cT "+to_string(cT.size())+")";
                // cout << "(cT " << cT.size() << ")";
            }
        }
        //接下来用来识别有数字的部分了，数字后面可能跟数字（数字常量）或者符号
        else if (isdigit(input[i])) {//如果是数字的话
            string num;
            while (i < input.size() && isdigit(input[i])) {//防止越界以及多位常数
                num += input[i];
                i++;
            }
            if(isalpha(input[i])|| input[i]== '.'){
                if(input[i]== '.'){
                    num+= input[i];
                    i++;
                    if(!isdigit(input[i])){
                        error_p= {Line, num+input[i], ERROR0};
                        //cout<<"错误类型：数字格式错误"<<endl;
                        return ERROR0;//数字格式错误，‘.’后必跟数字
                    }
                    while (i < input.size() && isdigit(input[i])) {//防止越界以及多位常数
                        num += input[i];
                        i++;
                    }
                }
                if(input[i]== 'E'||input[i]== 'e'){
                    num+= input[i];
                    i++;
                    if(input[i]== '+'){
                        num+= input[i];
                        i++;
                    }
                    if(!isdigit(input[i])){
                        error_p= {Line, num+input[i], ERROR0};
                        //cout<<"错误类型：数字格式错误";
                        return ERROR0;//数字格式错误，'e'后必跟数字
                    }
                    while (i < input.size() && isdigit(input[i])) {//防止越界以及多位常数
                        num += input[i];
                        i++;
                    }
                }else{
                    error_p= {Line, num+input[i], ERROR0};
                    //cout<<"错误类型：数字格式错误";
                    return ERROR0;//数字格式错误，包含了非法字符或标识符非法
                }
                
                
            }
            int judge= 0;
            
            for(int e=0; e<CT.size(); e++){//检测是否有重复
                if(stod(num)== CT[e]){
                    //cout << "(CT " << e+1 << ")";
                    Token.push_back({"CT", {e}});
                    line.push_back(Line);
                    //Token+= "(CT "+to_string(e+1)+")";
                    judge= 1;
                    break;
                }
            }
            if(judge== 0){
                CT.push_back(stod(num));
                line.push_back(Line);
                Token.push_back({"CT", {static_cast<int>(CT.size()-1)}});
                //Token+= "(CT "+to_string(CT.size())+")";
                //cout << "(CT " << CT.size() << ")";
            }
        }
        else if (fuhao(input[i])) {
            if (i + 1 < input.size()) {//防止数组越界访问
                string current= "";
                current += input[i];//验证可能有两个符号在一起
                current += input[i+1];
                if (current == ">=" ) {
                    Token.push_back({"PT", {8}});
                    line.push_back(Line);
                    //Token+= "(P 9)";
                    i += 2;
                    continue;
                }else if (current == "==" ) {
                    Token.push_back({"PT", {9}});
                    line.push_back(Line);
                    //Token+= "(P 10)";
                    i += 2;
                    continue;
                }else if (current == "<=" ) {
                    Token.push_back({"PT", {6}});
                    line.push_back(Line);
                    //Token+= "(P 7)";
                    i += 2;
                    continue;
                }
                // }else{
                //     if(fuhao(input[i+1])){
                //         cout<<i<<" 3";
                //         return 0;//界符错误
                //     }
                // }
                    
                
            }
            switch(input[i]) {
                case '=': Token.push_back({"PT", {0}});line.push_back(Line); break;
                case '+': Token.push_back({"PT", {1}});line.push_back(Line); break;
                case '-': Token.push_back({"PT", {2}});line.push_back(Line); break;
                case '*': Token.push_back({"PT", {3}});line.push_back(Line); break;
                case '/': Token.push_back({"PT", {4}});line.push_back(Line); break;
                case '<': Token.push_back({"PT", {5}});line.push_back(Line); break;
                case '>': Token.push_back({"PT", {7}});line.push_back(Line); break;
                case ':': Token.push_back({"PT", {10}});line.push_back(Line); break;
                case ';': Token.push_back({"PT", {11}});line.push_back(Line); break;
                case '(': Token.push_back({"PT", {12}});line.push_back(Line); break;
                case ')': Token.push_back({"PT", {13}});line.push_back(Line); break;
                case '.': Token.push_back({"PT", {14}});line.push_back(Line); break;
                case ',': Token.push_back({"PT", {15}});line.push_back(Line); break;
                case '[': Token.push_back({"PT", {16}});line.push_back(Line); break;
                case ']': Token.push_back({"PT", {17}});line.push_back(Line); break;
            }
            i++;
        }else if(input[i]== '\n'|| input[i]== ' '){
            if(input[i]== '\n'){
                Line++;
                //line.push_back(Line);
            }
            i++;
            continue;
        }else {
            if(input[i]== '.'){//防止对结尾识别异常
                break;
            }
            cout<<"错误类型：非法符号";
            //cout<<i<<" 4";
            string s(1, input[i]);//将字符转化为字符串存储
            error_p= {Line, s, ERROR0};
            return ERROR0;  //非法符号，例如@，￥等等
            
            
        }
    }
    for(int i= 0; i<Token.size(); i++){
        cout<<Token[i].first<<" "<<Token[i].second<<" ";
        
    }
    cout<<endl;
    for(int i: line){
        cout<<i<<" ";
    }
    //cout<<Token;
    // cout << "\nI :";
    // for (int i= 0; i<SYNBL.size(); i++){
    //     string s1= SYNBL[i].NAME;
    //     cout << s1 << " ";
    // } 
    // cout << "\nC :";
    // for (int i= 0; i< CT.size(); i++) {
    //     int num= CT[i];
    //     cout << num << " ";
    // }
    
    return NOERROR;//程序正常运行
}