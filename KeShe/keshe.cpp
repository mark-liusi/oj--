#include<bits/stdc++.h>
using namespace std;
vector<vector<string>> read(string filename)
{
    vector<vector<string>> content;
    ifstream file(filename);
    if(!file.is_open()){
        cout<<"error";
        return content;
    }

    string line;
    while(getline(file, line)){
        vector<string> row;
        stringstream cut(line);
        string cell;

        while (getline(cut, cell, ',')) {
            row.push_back(cell);
        }
        content.push_back(row);
    }
    return content;
}
pair<string, int> judge(const vector<vector<string>> content, vector<vector<int>> &dp, vector<vector<int>> &maze, int curx, int cury){
    int dx[4]= {-1, 0, 1, 0};
    int dy[4]= {0, -1, 0, 1};

    pair<string, int> final;//这个函数运行到最终的状态

    int count= 0;
    vector<vector<int>> next;
    for(int i=0; i<4; i++){
        int x= curx+dx[i], y= cury+dy[i];
        if(x>= 0&& x<dp.size()&& y>= 0&& y<dp[0].size()&& content[x][y]!= "#"&& maze[x][y]== -1){
            count++;
            next.push_back({x, y});
            maze[x][y]= 0;
        }
    }

    
    if(count== 0){//无路可走
        int cx= next[0][0], cy= next[0][1];
        maze[cx][cy]= 0;
        return {"noway", dp[cx][cy]};
    }else if(count> 0){
        int max= 0;
        for(int i= 0; i<count; i++){
            int cx= next[i][0], cy= next[i][1];
            pair<string, int> current= judge(content, dp, maze, cx, cy);
            final.first= current.first;
            if(current.second<= 0){//遇到陷阱
                if(current.first== "noway"){//在陷阱这条路上没有出路，所以避免金币数减少
                    
                }else{
                    dp[cx][cy]+= current.second;//出路在包含陷阱这条路上，只能从这里走
                }
            }else{
                max+= dp[cx][cy];
            }

            
        }
        dp[curx][cury]+= max;
        final.second= dp[curx][cury];
    }
    return final;
}
int main()
{
    map<string,int> Value = {{"S",0},{"",0},{"G",5},{"T",-3},{"L",0},{"B",0}};

    //cout<<Value["T"];
    string filename= "maz.csv";
    vector<vector<string>> content= read(filename);
    pair<int, int> start= {0, 0};\
    vector<vector<int>> maze(content.size(), vector<int>(content[0].size(), 0));
    vector<vector<int>> dp(content.size(), vector<int>(content[0].size(), 0));
    for(int i= 0; i<content.size(); i++){
        for(int j= 0; j<content[i].size(); j++){
            if(content[i][j]== "S"){
                start.first= i, start.second= j;
                maze[i][j]= -1;
            }else if(content[i][j]== "#"){
                maze[i][j]= 1;
                dp[i][j]= -1e9;
            }else if(content[i][j]== "T"){
                dp[i][j]= -3;
                maze[i][j]= -1;
            }else if(content[i][j]== "G"){
                dp[i][j]= 5;
                maze[i][j]= -1;
            }else if(content[i][j]== "L"|| content[i][j]=="B"||content[i][j]==" "){
                dp[i][j]= 0;
                maze[i][j]= -1;
            }
            cout<<content[i][j]<<" ";
        }
        cout<<endl;
    }


    for(int i= start.first; i<content[i].size(); i++){

    }
    //cout<<start.first<<" "<<start.second;
    return 0;
}