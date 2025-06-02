#include<iostream>
#include<vector>
using namespace std;
int judge(int min, int max, vector<int> weight, int numebr_of_cars);
int main()
{
    int number_of_package;//货物的数量
    int number_of_cars;//货车的数量
    int max= 0;//所有货物的装载量，作为最大的边界
    cin>>number_of_package>>number_of_cars;
    vector<int> weight(number_of_package);
    for(int i=0; i<number_of_package; i++){
        cin>>weight[i];
        max+= weight[i];
    }
    int loading = judge(0, max, weight, number_of_cars);//最小的最小装载量
    cout<<loading;
    return 0;

}
int judge(int min, int max, vector<int> weight, int numebr_of_cars)
{
    //cout<<weight.size();
    int judgement= 1;//用来判断车是能够继续装还是不能继续装
    int loading_capacity= (min+max)/ 2;//通过二分法来确定当前的装载量
    vector<int> current_capacity(numebr_of_cars);//存储每辆车当前的装载量
    int current_car= 0;//当前车的编号
    for(int i=0; i<weight.size(); i++){
        current_capacity[current_car]+= weight[i];
        if(current_capacity[current_car]> loading_capacity){
            current_car++;
            i--;
            continue;
        }
        if(current_car>= numebr_of_cars){//车不够用，也就是最大装载量不够
            judgement= 0;
            break;
        }
    }
    if(judgement== 0){
        if(max- min== 2){
            return judge(loading_capacity, max+1, weight, numebr_of_cars);
        }else if(max- min== 1){
            return max;
        }else{
            return judge(loading_capacity, max, weight, numebr_of_cars);
        }
        
    }else{
        // if(max- min== 2){
        //     return loading_capacity;
        // }
        return judge(min, loading_capacity, weight, numebr_of_cars);
    }
    return  loading_capacity;
}