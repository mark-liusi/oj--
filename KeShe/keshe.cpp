#include<bits/stdc++.h>
using namespace std;
vector<pair<int, int>> road;
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
pair<string, int> judge(const vector<vector<string>> content, vector<vector<int>> &dp, vector<vector<int>> &maze, int curx, int cury, pair<int, int> end, vector<pair<int, int>> &path){
    if(curx==end.first&& cury== end.second){
        return {"wayout", dp[curx][cury]};
    }
    int dx[4]= {-1, 0, 1, 0};
    int dy[4]= {0, -1, 0, 1};

    pair<string, int> final= {"noway", dp[curx][cury]};//这个函数运行到最终的状态

    int count= 0;
    vector<vector<int>> next;
    for(int i=0; i<4; i++){
        int x= curx+dx[i], y= cury+dy[i];
        if(x>= 0&& x<dp.size()&& y>= 0&& y<dp[0].size()){
            string temp= content[x][y];
            if(temp== "#"|| maze[x][y]!= -1){
                continue;
            }
            count++;
            next.push_back({x, y});
        }
    }

    int max= 0;
    int way= 0;
    vector<pair<int, int>> outpath;//走出去的主路
    vector<pair<int, int>> goldpath;//走不出去，但是收益为正的路
    if(count== 0){//无路可走
        return{"noway", dp[curx][cury]};
    }else if(count> 0){
        for(int i= 0; i<count; i++){
            int cx= next[i][0], cy= next[i][1];
            maze[cx][cy]= 0;
            vector<pair<int, int>> temppath;
            pair<string, int> current= judge(content, dp, maze, cx, cy, end, temppath);
            maze[cx][cy]= -1;
            int zou= 0;//该不该走
            //final.first= current.first;
            if(current.first== "wayout"){
                way= 1;
                zou= 1;
            }
            if(current.first== "noway"&& current.second> 0){//遇到陷阱
                zou= 1;
            }
            if(zou== 1){
                //road.push_back({cx, cy});
                max+= dp[cx][cy];
                if (current.first == "wayout" && outpath.empty()) {
                    outpath = temppath;
                }else {
                    goldpath.push_back({curx, cury});
                    goldpath.insert(goldpath.end(), temppath.begin(), temppath.end());
                    goldpath.push_back({curx, cury});
                }
            }
            
        }
        //dp[curx][cury]+= max;
        
    }
    dp[curx][cury]+= max;
    path.push_back({curx, cury});
    path.insert(path.end(), goldpath.begin(), goldpath.end());
    path.insert(path.end(), outpath.begin(), outpath.end());

    if(way== 1){
        final.first= "wayout";
    }
    final.second= dp[curx][cury];
    return final;
}
int main()
{
    map<string,int> Value = {{"S",0},{" ",0},{"G",5},{"T",-3},{"L",0},{"B",0}, {"E", 0}};

    //cout<<Value["T"];
    string filename= "maz.csv";
    vector<vector<string>> content= read(filename);
    pair<int, int> start= {0, 0};
    pair<int, int> end= {0, 0};
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
            }else if(content[i][j]== "E"){
                dp[i][j]= 0;
                maze[i][j]= -1;
                end.first= i, end.second= j;
            }
            //cout<<content[i][j]<<" ";
        }
        //cout<<endl;
    }

    vector<pair<int, int>> path;
    pair<string, int> result= judge(content, dp, maze, start.first, start.second, end, path);
    cout<<result.first<<" "<<result.second<<endl;
    // for (int i= 0; i<path.size(); i++) {
    //     cout << "(" << path[i].first << "," << path[i].second << ")";
    // }
    vector<pair<int, int>> finalPath;
    for (int i= 0; i< path.size(); i++) {
        if (i== 0 || path[i]!= path[i - 1]) {
            finalPath.push_back(path[i]);
        }
    }

    for (auto p : finalPath) {
        cout << "(" << p.first << "," << p.second << ")";
    }
    //cout<<start.first<<" "<<start.second;
    return 0;
}