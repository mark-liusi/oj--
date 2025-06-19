#include<iostream>
using namespace std;
class Person {
public:
    void Print() {
        cout << "name:" << _name << endl;
    }
protected:
    string _name = "LiMing";
private:
    int _age;
};

class Student : public Person {
public:
    void Set(const char* name, int age) {
        _name = name;  // 正确访问protected成员
        // _age = age;  // 错误:不能访问private成员
    }
protected:
    int _stid;
};

int main() {
    Student s;
    s.Set("张三", 20);
    s.Print();  // 输出: name:张三
    return 0;
}