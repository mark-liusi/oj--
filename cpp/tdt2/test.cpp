#include<bits/stdc++.h>
using namespace std;
class Person{
    public:
    void print()
    {
        cout << "name:" << _name << endl;
		cout << "age:" << _age << endl;
    }
    protected:
	    string _name = "LiMing";  //姓名
	private:
        int _age;
};
class stu: private Person{
    public:
    void Set(string name, int age){
        _name= name;
        //_age= age;
    }
};

int main()
{

}