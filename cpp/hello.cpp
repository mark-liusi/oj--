#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
using namespace std;

void printout(const vector<vector<double>>& a) {
    int n = a.size();
    int m = a[0].size();
    cout << fixed << setprecision(4);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (j == m-1) cout << " | ";
            cout << setw(8) << a[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void printVector(const vector<double>& x) {
    cout << fixed << setprecision(4);
    cout << "[";
    for (int i = 0; i < x.size(); i++) {
        cout << x[i];
        if (i != x.size()-1) cout << ", ";
    }
    cout << "]" << endl;
}

void gauss(vector<vector<double>> A, vector<double> b) {
    int n = A.size();
    vector<vector<double>> aug(n, vector<double>(n+1));
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) aug[i][j] = A[i][j];
        aug[i][n] = b[i];
    }

    cout << "初始增广矩阵：" << endl;
    printout(aug);

    for (int k = 0; k < n; k++) {
        int p = k;
        double maxVal = abs(aug[k][k]);
        for (int i = k+1; i < n; i++) {
            if (abs(aug[i][k]) > maxVal) {
                maxVal = abs(aug[i][k]);
                p = i;
            }
        }

        if (p != k) {
            swap(aug[k], aug[p]);
            cout << "交换行" << k+1 << "和行" << p+1 << "后的增广矩阵：" << endl;
            printout(aug);
        }

        for (int i = k+1; i < n; i++) {
            if (aug[k][k] == 0) {
                cout << "矩阵中主元为0，无法继续进行消元!" << endl;
                return;
            }
            double factor = aug[i][k] / aug[k][k];
            for (int j = k; j <= n; j++) {
                aug[i][j] -= factor * aug[k][j];
            }
        }

        cout << "第" << k+1 << "步消元后的增广矩阵：" << endl;
        printout(aug);
    }

    vector<double> x(n);
    for (int i = n-1; i >= 0; i--) {
        x[i] = aug[i][n];
        for (int j = i+1; j < n; j++) {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    cout << "列主元高斯消去法解为：" << endl;
    printVector(x);
}

// Gauss-Seidel迭代法
void gaussSeidel(vector<vector<double>> A, vector<double> b, int iterations) {
    int n = A.size();
    vector<double> x(n, 0.0);
    cout << "初始解：[0.0000, 0.0000, 0.0000]" << endl;

    for (int k = 0; k < iterations; k++) {
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) sum += A[i][j] * x[j];
            for (int j = i+1; j < n; j++) sum += A[i][j] * x[j];
            x[i] = (b[i] - sum) / A[i][i];
        }
        cout << "第" << k+1 << "次迭代结果：";
        printVector(x);
    }
}

void SOR(vector<vector<double>> A, vector<double> b, double omega, int iterations) {
    int n = A.size();
    vector<double> x(n, 0.0);
    cout << "初始解：[0.0000, 0.0000, 0.0000]" << endl;

    for (int k = 0; k < iterations; k++) {
        vector<double> x_new = x;
        for (int i = 0; i < n; i++) {
            double sum1 = 0.0, sum2 = 0.0;
            for (int j = 0; j < i; j++) sum1 += A[i][j] * x_new[j];
            for (int j = i+1; j < n; j++) sum2 += A[i][j] * x[j];
            x_new[i] = (1 - omega) * x[i] + omega * (b[i] - sum1 - sum2) / A[i][i];
        }
        x = x_new;
        cout << "第" << k+1 << "次迭代结果：";
        printVector(x);
    }
}

int main() {
    
    int hang= 0;
    int lie= 0;
    cin>>hang>>lie;
    vector<vector<double>> A(hang, vector<double>(lie));
    for(int i=0; i<hang; i++){
        for(int j=0; j<lie; j++){
            cin>>A[i][j];
        }
    }
    int jie= 0;
    cin>>jie;
    vector<double> b(jie) ;
    for(int i=0; i<jie; i++){
        cin>>b[i];
    }

    cout << "列主元高斯消去法" << endl;
    gauss(A, b);
    cout << endl;

    cout << "Gauss-Seidel ===" << endl;
    gaussSeidel(A, b, 3);
    cout << endl;

    cout << "SOR迭代法" << endl;
    SOR(A, b, 1.2, 2);

    return 0;
}
