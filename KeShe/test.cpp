#include <bits/stdc++.h>
using namespace std;

// 最终路径
vector<pair<int, int>> road;

// 读取 CSV 文件
vector<vector<string>> read(string filename) {
    vector<vector<string>> content;
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "error opening file" << endl;
        return content;
    }

    string line;
    while (getline(file, line)) {
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

// 主逻辑
pair<string, int> judge(const vector<vector<string>> &content, vector<vector<int>> &dp, vector<vector<int>> &maze, int curx, int cury, pair<int, int> end, vector<pair<int, int>> &path)
{
    if (curx== end.first&& cury== end.second) {
        path.push_back({curx, cury});
        return {"wayout", dp[curx][cury]};
    }
    int dx[4]= {-1, 0, 1, 0};
    int dy[4]= {0, -1, 0, 1};

    int totalGain= 0;
    int hasWayOut= 0;

    vector<pair<int, int>> mainPath;
    vector<pair<int, int>> gatherPath;

    for (int i= 0; i< 4; i++) {
        int x= curx+ dx[i], y= cury+ dy[i];
        if (x< 0 || x>= dp.size() || y< 0 || y>= dp[0].size()) continue;

        string cell = content[x][y];

        if (cell == "#" || maze[x][y] != -1) continue;

        maze[x][y] = 0;
        vector<pair<int, int>> subPath;
        pair<string, int> res = judge(content, dp, maze, x, y, end, subPath);
        maze[x][y] = -1;

        bool shouldWalk = false;
        if (res.first == "wayout") {
            hasWayOut = true;
            shouldWalk = true;
        } else if (res.first == "noway" && res.second > 0) {
            shouldWalk = true;
        }

        if (shouldWalk) {
            totalGain += dp[x][y];
            if (res.first == "wayout" && mainPath.empty()) {
                mainPath = subPath;
            } else {
                gatherPath.push_back({curx, cury});
                gatherPath.insert(gatherPath.end(), subPath.begin(), subPath.end());
                gatherPath.push_back({curx, cury});
            }
        }
    }

    dp[curx][cury] += totalGain;
    path.push_back({curx, cury});
    path.insert(path.end(), gatherPath.begin(), gatherPath.end());
    path.insert(path.end(), mainPath.begin(), mainPath.end());

    pair<string, int> final = {"noway", dp[curx][cury]};
    if (hasWayOut) final.first = "wayout";
    final.second = dp[curx][cury];
    return final;
}

int main() {
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
    pair<string, int> result = judge(content, dp, maze, start.first, start.second, end, path);

    cout << result.first<<" "<<result.second<<endl;
    for (int i= 0; i<path.size(); i++) {
        cout << "(" << path[i].first << "," << path[i].second << ") ";
    }
    cout << endl;

    return 0;
}
