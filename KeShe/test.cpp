#include <iostream>
#include <vector>
using namespace std;

// 生成全排列的递归函数
// list: 存储当前排列的数组
// k: 当前处理的起始位置
// m: 数组的最后一个位置
void Perm(vector<int>& list, int k, int m) {
    // 当处理到最后一个元素时，输出当前排列
    if (k == m) {
        cout << "  找到一个排列: ";
        for (int i = 0; i <= m; i++) {
            cout << list[i] << " ";
        }
        cout << endl;
    } 
    else {
        // 遍历从当前位置到末尾的所有元素
        for (int i = k; i <= m; i++) {
            swap(list[k], list[i]);
            Perm(list, k + 1, m);
            swap(list[k], list[i]);
        }
    }
}

int main() {
    int n;
    cout << "请输入要生成全排列的元素数量 (1-5): ";
    cin >> n;
    
    if (n < 1 || n > 5) {
        cout << "输入范围应为1-5（为了输出清晰）" << endl;
        return 1;
    }
    
    // 创建初始序列 [1, 2, ..., n]
    vector<int> arr;
    for (int i = 1; i <= n; i++) {
        arr.push_back(i);
    }
    
    cout << "\n开始生成 " << n << " 个元素的全排列:\n";
    cout << "初始序列: ";
    for (int num : arr) cout << num << " ";
    cout << "\n" << endl;
    
    Perm(arr, 0, n-1);
    
    return 0;
}