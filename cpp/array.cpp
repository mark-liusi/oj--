#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <limits>

using namespace std;

class CubicSpline {
private:
    vector<double> x, y;
    vector<double> a, b, c, d;
    int n;

public:
    CubicSpline(const vector<double>& x_data, const vector<double>& y_data) 
        : x(x_data), y(y_data) {
        n = x.size() - 1;
        a = y;
        b.resize(n, 0.0);
        c.resize(n + 1, 0.0);
        d.resize(n, 0.0);

        vector<double> h(n);
        for (int i = 0; i < n; i++) {
            h[i] = x[i + 1] - x[i];
        }

        vector<double> alpha(n);
        for (int i = 1; i < n; i++) {
            alpha[i] = 3.0 * ((a[i + 1] - a[i]) / h[i] - (a[i] - a[i - 1]) / h[i - 1]);
        }

        vector<double> l(n + 1, 1.0), mu(n + 1, 0.0), z(n + 1, 0.0);
        l[0] = 1.0;
        z[0] = 0.0;
        mu[0] = 0.0;
        c[0] = 0.0;

        for (int i = 1; i < n; i++) {
            l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }

        l[n] = 1.0;
        z[n] = 0.0;
        c[n] = 0.0;

        for (int j = n - 1; j >= 0; j--) {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3.0;
            d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
        }
    }

    double evaluate(double t) {
        if (t < x[0] || t > x[n]) return NAN;

        int i = distance(x.begin(), upper_bound(x.begin(), x.end(), t)) - 1;
        i = max(0, min(i, n - 1));

        double dx = t - x[i];
        return a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx;
    }

    string get_segment(int i) {
        if (i < 0 || i >= n) return "";

        stringstream ss;
        ss << fixed << setprecision(6);
        ss << "S_" << i << "(x) = " << a[i] << " + " 
           << b[i] << "(x - " << x[i] << ") + " 
           << c[i] << "(x - " << x[i] << ")^2 + " 
           << d[i] << "(x - " << x[i] << ")^3";
        return ss.str();
    }

    void print_segments() {
        for (int i = 0; i < n; i++) {
            cout << "\u533a\u95f4 [" << x[i] << ", " << x[i+1] << "]:\n";
            cout << get_segment(i) << "\n\n";
        }
    }
};

void plot_in_terminal(const vector<double>& x, const vector<double>& y, CubicSpline& spline) {
    const int TERM_WIDTH = 80;
    const int TERM_HEIGHT = 20;
    const double MARGIN = 0.5;

    double min_depth = *min_element(y.begin(), y.end()) - MARGIN;
    double max_depth = *max_element(y.begin(), y.end()) + MARGIN;
    double depth_range = max_depth - min_depth;

    vector<vector<char>> plot(TERM_HEIGHT, vector<char>(TERM_WIDTH, ' '));

    int x_axis_row = static_cast<int>((max_depth - 0.0) * (TERM_HEIGHT - 1) / depth_range);
    if (x_axis_row >= 0 && x_axis_row < TERM_HEIGHT) {
        for (int col = 0; col < TERM_WIDTH; col++) {
            plot[x_axis_row][col] = '-';
        }
    }

    for (int col = 0; col < TERM_WIDTH; col++) {
        double x_val = col * 20.0 / TERM_WIDTH;
        double depth = spline.evaluate(x_val);
        if (!isnan(depth)) {
            int row = static_cast<int>((max_depth - depth) * (TERM_HEIGHT - 1) / depth_range);
            row = max(0, min(TERM_HEIGHT - 1, row));
            plot[row][col] = '.';
        }
    }

    for (int i = 0; i < x.size(); i++) {
        int col = static_cast<int>(x[i] * TERM_WIDTH / 20.0);
        col = max(0, min(TERM_WIDTH - 1, col));
        int row = static_cast<int>((max_depth - y[i]) * (TERM_HEIGHT - 1) / depth_range);
        row = max(0, min(TERM_HEIGHT - 1, row));
        plot[row][col] = '*';
    }

    cout << "\n\u6c9f\u5e95\u5730\u5f62\u56fe (Y\u8f74:\u6df1\u5ea6, X\u8f74:\u8ddd\u79bb):\n";
    cout << "  *: \u539f\u59cb\u6570\u636e\u70b9\n";
    cout << "  .: \u62df\u5408\u66f2\u7ebf\n\n";

    for (int row = 0; row < TERM_HEIGHT; row++) {
        cout << setw(3) << fixed << setprecision(1) 
             << (max_depth - row * depth_range / (TERM_HEIGHT - 1)) << " | ";
        for (int col = 0; col < TERM_WIDTH; col++) {
            cout << plot[row][col];
        }
        cout << endl;
    }

    cout << "    ";
    for (int col = 0; col < TERM_WIDTH; col++) {
        if (col % 8 == 0) cout << "+";
        else cout << " ";
    }
    cout << "\n    ";
    for (int col = 0; col < TERM_WIDTH; col += 8) {
        double x_val = col * 20.0 / TERM_WIDTH;
        cout << setw(4) << fixed << setprecision(0) << x_val;
    }
    cout << "\n";
}

int main() {
    vector<double> x = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    vector<double> y = {9.01, 8.96, 7.96, 7.96, 8.02, 9.05, 10.13, 11.18, 12.26, 13.28,
                        13.32, 12.61, 11.29, 10.22, 9.15, 7.95, 7.95, 8.86, 9.81, 10.80, 10.93};

    double total_length = 0.0;
    for (int i = 0; i < x.size() - 1; i++) {
        double dx = x[i+1] - x[i];
        double dy = y[i+1] - y[i];
        total_length += sqrt(dx*dx + dy*dy);
    }

    cout << "\u95ee\u9898(1) \u5149\u7f00\u957f\u5ea6\u8fd1\u4f3c\u503c: " << fixed << setprecision(2) 
         << total_length << " \u7c73\n\n";

    CubicSpline spline(x, y);

    cout << "\u95ee\u9898(2) \u62df\u5408\u66f2\u7ebf\u51fd\u6570(\u5206\u6bb5\u4e09\u6b21\u6837\u6761):\n";
    spline.print_segments();

    plot_in_terminal(x, y, spline);

    return 0;
}
