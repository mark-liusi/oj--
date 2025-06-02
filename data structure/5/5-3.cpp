#include<iostream>
#include<cmath>
#include<vector>
#include<iomanip>
using namespace std;
typedef struct zuobiao
{
    double x;
    double y;
}zuobiao;

typedef struct xian
{
    zuobiao a;
    zuobiao b;
}xian;
void koch1(xian a, vector<xian> &e);
int main()
{
    int number;
    cin>>number;
    vector<xian> c(1);
    c[0].a.x= 0.0, c[0].a.y= 0.0;
    c[0].b.x= 100.0, c[0].b.y= 0.0;

    int count= 0;
    int k= 0;
    while(k<number){
        vector<xian> d(pow(4, k));
        d.assign(c.begin(), c.end());
        c.clear();
        for(xian i: d){
            koch1(i, c);
        }
        d.clear();
        k++;
    }
    for(xian i: c){
        cout<<fixed<<setprecision(8)<<i.a.x<<" "<<i.a.y<<endl;
    }
    cout<<fixed<<setprecision(8)<<c.back().b.x<<" "<<c.back().b.y<<endl;
    return 0;
}
void koch1(xian a, vector<xian> &e)
{
    zuobiao s, t, u;

    // 三等分点 s (1/3) 和 t (2/3)
    s.x = (2*a.a.x + a.b.x) / 3.0;
    s.y = (2*a.a.y + a.b.y) / 3.0;
    t.x = (a.a.x + 2*a.b.x) / 3.0;
    t.y = (a.a.y + 2*a.b.y) / 3.0;

    // 计算 t - s 的向量
    double dx = t.x - s.x;
    double dy = t.y - s.y;

    // 旋转60度 (π/3)，构造出顶点 u
    double angle = M_PI / 3.0;
    u.x = s.x + dx * cos(angle) - dy * sin(angle);
    u.y = s.y + dx * sin(angle) + dy * cos(angle);

    // 构造4条边
    e.push_back({a.a, s});
    e.push_back({s, u});
    e.push_back({u, t});
    e.push_back({t, a.b});
}
