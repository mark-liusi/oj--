#include <iostream>
#include <vector>
using namespace std;
int judge(vector<int> weight, int number_of_cars, int sum, int max) {
    // 初始二分范围：最重的包裹到所有包裹的总重量
    int right= sum;
    int left= max;

    while (left < right) {
        int mid = (left + right) / 2;

        int car_count = 1;  // 当前用了多少辆车
        int current_load = 0;  // 当前这辆车的载重

        for (int w : weight) {
            if (current_load + w <= mid) {
                current_load += w;
            } else {
                car_count++;
                current_load = w;
            }
        }

        if (car_count <= number_of_cars) {
            // 可以装下，尝试降低载重
            right = mid;
        } else {
            // 装不下
            left = mid + 1;
        }
    }

    return left;  // 此时 left == right，就是最小可行载重
}

int main() {
    int number_of_package; // 货物数量
    int number_of_cars;    // 货车数量
    cin >> number_of_package >> number_of_cars;

    vector<int> weight(number_of_package);
    int sum= 0;
    int max= 0;
    for (int i = 0; i < number_of_package; i++) {
        cin >> weight[i];
        if(weight[i]> max){
            max= weight[i];//所有包裹中最重的包裹重量
        }
        sum+= weight[i];//所有包裹的总重量
    }

    int loading = judge(weight, number_of_cars, sum, max);
    cout << loading << endl;

    return 0;
}
