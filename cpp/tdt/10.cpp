#include<iostream>
#include<vector>
#include<cmath>
using namespace std;
int judge(vector<vector<int>> &a, int number){
    int repeat= 0;
    for(int i= 0; i<number; i++){//行检测
        for(int j= 0; j<number; j++){
            if(a[i][j]== 1){
                repeat++;
            }
            if(repeat== 2){
                return 0;
            }
        }
    }
    repeat= 0;
    for(int i= 0; i<number; i++){//列检测
        for(int j= 0; j<number; j++){
            if(a[j][i]== 1){
                repeat++;
            }
            if(repeat== 2){
                return 0;
            }
        }
    }

    int i= 0;//斜检测
    repeat= 0;
    while (i<number)
    {
        if(a[i][i]== 1){
            repeat++;
        }
        if(repeat== 2){
            return 0;
        }
        i++;
    }
    repeat= 0;
    i= 1;
    for(int i= 1; i<number; i++){
        int j= i-1;
        if(a[i][j]== 1){
            repeat++;
        }
        if(repeat== 2){
            return 0;
        }
    }
    i= 1, repeat= 0;
    for(int i= 1; i<number-1; i++){
        int j= i+1;
        if(a[i][j]== 1){
            repeat++;
        }
        if(repeat== 2){
            return 0;
        }
    }
    return 1;
}

int main()
{
    int number;
    cin>>number;
    vector<vector<int>> a(number, vector<int>(number, 0));
    vector<long long> b;
    for(long long i= 0; i<pow(2, number*number); i++){
        int position= 0;
        long long temp= i;
        while(temp!= 0){
            if((1&temp)== 1){
                position++;
            }
            temp= temp>>1;
        }
        if(position== number){
            b.push_back(i);
        }
    }
    vector<long long> d;
    int count= 0;
    for(int i: b){
        vector<vector<int>> c= a;
        int m= 0, n= 0;
        while(i!= 0){
            if(1&i== 1){
                c[m][n]= 1;
                if(m== 5){
                    m++;
                }
                n++;
            }
            i= i>>1;
        }
        if(judge(c, number)){
            d.push_back(i);
            count++;
        }
    }
    for(int j= 0; j<3; j++){
        vector<vector<int>> c= a;
        int m= 0, n= 0;
        long long i= d[j];
        while(i!= 0){
            if(1&i== 1){
                c[m][n]= 1;
                if(m== 5){
                    m++;
                }
                n++;
            }
            i= i>>1;
        }
        for(int h= 0; h<number; h++){
            for(int y=0; y<number; y++){
                if(c[h][y]== 1){
                    cout<<y<<" ";
                }
            }
        }
        cout<<endl;
    }
    cout<<count;
    return 0;
}