#include <iostream>
using namespace std;

// 定义链表节点结构体
typedef struct todo {
    string name;
    int time;
    todo* next;
} todo;

int main() {
    // 初始化头节点（不存储实际数据）
    todo* head = new todo;
    head->next = nullptr;  // 头节点初始为空链表

    // 当前链表末尾指针
    todo* current = head;

    // 录入5个数据
    for (int i = 0; i < 5; i++) {
        // 创建新节点
        todo* newNode = new todo;
        cin >> newNode->name >> newNode->time;
        newNode->next = nullptr;

        // 将新节点链接到链表末尾
        current->next = newNode;
        current = newNode;  // 更新末尾指针
    }

    /*------------------------------------------------
      验证链表内容（可选）
    -------------------------------------------------*/
    cout << "链表内容：" << endl;
    todo* p = head->next;  // 跳过头节点
    while (p != nullptr) {
        cout << p->name << " " << p->time << endl;
        p = p->next;
    }

    /*------------------------------------------------
      释放内存（完整程序需要）
    -------------------------------------------------*/
    p = head->next;
    while (p != nullptr) {
        todo* temp = p;
        p = p->next;
        delete temp;
    }
    delete head;  // 释放头节点

    return 0;
}