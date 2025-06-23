#include<bits/stdc++.h>
using namespace std;
int counta= 0;
//int row= 0;
void huanghou(int number, int row, vector<int>& a,vector<int> &line, vector<int> &Zdui,vector<int> &Fdui){
    if(row== number){
        for(int i= 0; i<number; i++){
            cout<<a[i]+1<<" ";
        }
        cout<<endl;
        counta++;
        return;
    }

    for(int i= 0; i<number; i++){
        if(line[i]|| Zdui[number-(row-i)-1]|| Fdui[row+i]){
            continue;
        }else{
            a[row]= i;
            line[i]= 1, Zdui[number-(row-i)-1]= 1, Fdui[row+i]= 1;
            huanghou(number, row+1, a, line, Zdui, Fdui);
            a[row]= 0;
            line[i]= 0, Zdui[number-(row-i)-1]= 0, Fdui[row+i]= 0;
        }
    }
    return ;
    

}
int main(){
    int number;
    cin>>number;
    vector<int> a(number);
    vector<int> line(number, 0);
    vector<int> Zdui(2*number-1, 0);
    vector<int> Fdui(2*number-1, 0);
    huanghou(number, 0, a, line, Zdui, Fdui);
    cout<<counta;
    return 0;
}