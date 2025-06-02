#include<iostream>
#include<vector>
#include<set>
#include<cmath>
int zuhe(int degree, int number);
int jiecheng(int begin, int end);
using namespace std;
int main()
{
    int numebr_of_small;
    int number_of_sum;
    cin>>numebr_of_small;
    vector<int> a(pow(2, numebr_of_small));
    set<int> c;
    for(int i=0; i<numebr_of_small; i++){
        cin>>a[i];
        c.insert(a[i]);
    }

    

    int degree= 2;
    int j= 5;
    int current= 0;
    //cout<<zuhe(0, 6);
    while(degree<= numebr_of_small){
        
        //int position= 0;
        for(int i=1; i<zuhe(degree-1, numebr_of_small); i++){
            a[j]= a[current]+ a[current+i];
            c.insert(a[j]);
            j++;
            if(i== zuhe(degree-1, numebr_of_small)-1){
                if(i- current== 1){
                    break;
                }
                current++, i= 1;
            }
            
        }
    }
    cin>>number_of_sum;
    vector<int> b(number_of_sum);
    for(int i=0; i<number_of_sum; i++){
        cin>>b[i];
        if(c.find(b[i])!= c.end()){
            cout<<"yes"<<endl;
        }else{
            cout<<" no"<<endl;
        }
    }

    return 0;
}
int zuhe(int degree, int number)
{
    int down= jiecheng(1, number- degree);
    int up= jiecheng(degree+1, number);
    return up/down;

}
int jiecheng(int begin, int end)
{
    int sum= 1;
    for(int i= begin; i<= end; i++){
        sum*= i;
    }
    return sum;
}