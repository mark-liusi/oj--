#include<iostream>
#include<vector>
using namespace std;
int count= 0;
int repeat= 0;
void judge(int row, vector<int>& lie,  vector<int>& zhengdui, vector<int>& fandui, int number, vector<int> &l)
{
    if(row== number){
        count++;
        if(repeat< 3){
            for(int i= 0; i<number; i++){
                cout<<l[i]+1<<" ";
            }
            cout<<endl;
            repeat++;
        }
        
        return ;
    }
    for(int i= 0; i<number; i++){
        if(lie[i]|| zhengdui[i+row]|| fandui[i-row+number]){
            continue;
        }
        lie[i]= zhengdui[i+row]= fandui[i-row+number]= 1;
        l[row]= i;
        judge(row+1, lie, zhengdui, fandui, number, l);
        lie[i]= zhengdui[i+row]= fandui[i-row+number]= 0;
    }
}
int main()
{
    int number;
    cin>>number;
    vector<int> lie(number, 0);
    vector<int> zhengdui(number*2, 0);
    vector<int> fandui(number*2, 0);
    vector<int> l(number);
    judge(0, lie, zhengdui, fandui, number, l);
    cout<<count;
    return 0;

}