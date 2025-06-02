#include <iostream>
#include <vector>
#include <unordered_set>
#include <stack>
using namespace std;

int Partition(vector<int>& A, int p, int r, unordered_set<int>& pivots) {
    int x = A[r];
    int i = p - 1;
    for (int j = p; j <= r - 1; j++) {
        if (A[j] <= x) {
            i++;
            swap(A[i], A[j]);
        }
    }
    swap(A[i + 1], A[r]);
    int q = i + 1;
    pivots.insert(q);
    return q;
}

void QuickSort(vector<int>& A, int n, unordered_set<int>& pivots) {
    stack<pair<int, int>> stk;
    stk.push({0, n - 1});

    while (!stk.empty()) {
        auto [p, r] = stk.top();
        stk.pop();

        if (p >= r) continue;

        int q = Partition(A, p, r, pivots);
        stk.push({q + 1, r});
        stk.push({p, q - 1});
    }
}

int main() {
    int n;
    cin >> n;
    vector<int> A(n);
    for (int i = 0; i < n; i++) {
        cin >> A[i];
    }

    unordered_set<int> pivots;
    QuickSort(A, n, pivots);

    for (int i = 0; i < n; i++) {
        if (pivots.count(i)) {
            cout << "[" << A[i] << "]";
        } else {
            cout << A[i];
        }
        if (i != n - 1) {
            cout << " ";
        }
    }
    cout << endl;

    return 0;
}