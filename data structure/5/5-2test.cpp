#include <iostream>
#include <vector>
using namespace std;

// 定义哨兵值（理论上应为无穷大）
const int SENTINEL = 1e9; // 假设输入数据不会超过这个值

// 合并函数
void Merge(vector<int> &A, int left, int mid, int right)
{
    int n1 = mid - left;     // 左半部分长度
    int n2 = right - mid;    // 右半部分长度

    // 创建临时数组 L 和 R，并初始化大小为 n1+1 和 n2+1（包含哨兵）
    vector<int> L(n1 + 1);
    vector<int> R(n2 + 1);

    // 复制数据到临时数组
    for (int i = 0; i < n1; i++) {
        L[i] = A[left + i];
    }
    for (int i = 0; i < n2; i++) {
        R[i] = A[mid + i];
    }

    // 添加哨兵值
    L[n1] = SENTINEL;
    R[n2] = SENTINEL;

    // 合并两个数组
    int i = 0, j = 0;
    for (int k = left; k < right; k++) { // 注意：k 的范围是 [left, right-1]
        if (L[i] <= R[j]) {
            A[k] = L[i];
            i++;
        } else {
            A[k] = R[j];
            j++;
        }
    }
}

// 归并排序函数
void Merge_Sort(vector<int> &A, int left, int right)
{
    if (left + 1 < right) { // 子数组长度大于1时继续分割
        int mid = (left + right) / 2; // 计算中间位置
        Merge_Sort(A, left, mid);     // 对左半部分排序
        Merge_Sort(A, mid, right);    // 对右半部分排序
        Merge(A, left, mid, right);   // 合并两部分
    }
}

int main()
{
    int n;
    cin >> n; // 输入数组长度
    vector<int> A(n);
    for (int i = 0; i < n; i++) {
        cin >> A[i]; // 输入数组元素
    }

    // 调用归并排序
    Merge_Sort(A, 0, n);

    // 输出排序结果
    for (int i = 0; i < n; i++) {
        cout << A[i] << " ";
    }
    cout << endl;

    return 0;
}