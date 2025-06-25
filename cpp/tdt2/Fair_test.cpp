#include <bits/stdc++.h>
using namespace std;

const int MAX_STEPS = 60;

int solve(vector<long long>& a) {
    vector<double> logA(a.size());
    for (int i = 0; i < a.size(); i++) {
        logA[i] = log10(a[i]);
    }

    int ops = 0;
    for (int i = 1; i < a.size(); i++) {
        if (logA[i] >= logA[i - 1]) continue;

        int step = 0;
        double val = logA[i];

        while (val < logA[i - 1] && step <= MAX_STEPS) {
            val *= 2.0;
            step++;
        }

        if (step > MAX_STEPS) return -1;

        logA[i] = val;
        ops += step;
    }

    return ops;
}

int main() {
    int t;
    cin >> t;
    while (t--) {
        int n;
        cin >> n;
        vector<long long> a(n);
        for (int i = 0; i < n; i++) {
            cin >> a[i];
        }

        cout << solve(a) << '\n';
    }

    return 0;
}

