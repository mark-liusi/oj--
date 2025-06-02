
#include<iostream>
#include<vector>
using namespace std;

bool fuhao(char c) {
    string ops = "+-*/";
    return ops.find(c) != string::npos;
}

void zimushuzi(string input, vector<int> &position, vector<int> &judge, vector<int> &judgekuohao);
void fh(string input, vector<int> &position, vector<int> &judge, vector<int> &judgekuohao);

int main() {
    string input;
    cin >> input;
    vector<int> judge(1, 1);
    vector<int> judgekuohao(1, 0);
    vector<int> position(1, 0);

    if(input.empty() || (input[0] != '(' && !isalnum(input[0]))) {
        cout << "false";
        return 0;
    }

    if(input[position[0]] == '(') {
        judgekuohao[0] = 1;
        position[0]++;
        zimushuzi(input, position, judge, judgekuohao);
    } 
    else if(isalnum(input[position[0]])) {
        position[0]++;
        zimushuzi(input, position, judge, judgekuohao);
    }

    if(judge[0] && judgekuohao[0] == 0 && position[0] == input.size()) {
        cout << "true";
    } else {
        cout << "false";
    }
    return 0;
}

void zimushuzi(string input, vector<int> &position, vector<int> &judge, vector<int> &judgekuohao) {
    if(position[0] >= input.size()) {
        judge[0] = (judgekuohao[0] == 0) ? 1 : 0;
        return;
    }

    while(position[0] < input.size() && isalnum(input[position[0]])) {
        position[0]++;
    }
    if(position[0] >= input.size()) return;

    if(input[position[0]] == ')') {
        if(judgekuohao[0] <= 0) {
            judge[0] = 0;
            return;
        }
        judgekuohao[0]--;
        position[0]++;
        zimushuzi(input, position, judge, judgekuohao);
    }
    else if(fuhao(input[position[0]])) {
        fh(input, position, judge, judgekuohao);
    }
    else if(input[position[0]] != '(') {
        judge[0] = 0;
    }
}

void fh(string input, vector<int> &position, vector<int> &judge, vector<int> &judgekuohao) {
    position[0]++;
    if(position[0] >= input.size()) {
        judge[0] = 0;
        return;
    }

    if(input[position[0]] == '(') {
        judgekuohao[0]++;
        position[0]++;
        zimushuzi(input, position, judge, judgekuohao);
    }
    else if(isalnum(input[position[0]])) {
        position[0]++;
        zimushuzi(input, position, judge, judgekuohao);
    }
    else {
        judge[0] = 0;
    }
}
