#include<iostream>
using namespace std;
int main()
{
    int n;
    cin >> n;
    for(int i = 0; i < (1<<n); i++) //从0～2^n-1个状态
    {
        for(int j = 0; j < n; j++) //遍历二进制的每一位
        {
            if(i & (1 << j))//判断二进制第j位是否存在
            {
                printf("%d ",j);//如果存在输出第j个元素
            }
        }
        printf("\n");
    }
    return 0;
}