#include <iostream>
#include <vector>
using namespace std;

bool canSum(int index, int currentSum, int target, const vector<int>& a) {
    if (currentSum == target) return true;
    if (index >= a.size()) return false;
    // 选择当前元素
    if (currentSum + a[index] <= target) {
        if (canSum(index + 1, currentSum + a[index], target, a)) return true;
    }
    // 不选择当前元素
    return canSum(index + 1, currentSum, target, a);
}

int main() {
    int n;
    cin >> n;
    vector<int> a(n);
    for (int i = 0; i < n; ++i) {
        cin >> a[i];
    }
    int q;
    cin >> q;
    for (int i = 0; i < q; ++i) {
        int m;
        cin >> m;
        bool found = canSum(0, 0, m, a);
        cout << (found ? "yes" : "no") << endl;
    }
    return 0;
}