#include <GL/glut.h>
#include <cmath>

// 窗口大小
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;

void display() {
    // 清屏
    glClear(GL_COLOR_BUFFER_BIT);
    
    // 绘制一行不同大小的点
    int numPoints = 10;  // 点的数量
    float startX = -0.8f;  // 起始 x 坐标
    float spacing = 1.6f / (numPoints - 1);  // 点之间的间距
    float y = 0.0f;  // y 坐标(水平线)
    
    for (int i = 0; i < numPoints; i++) {
        float x = startX + i * spacing;
        
        // 设置点的大小(从小到大)
        float pointSize = 2.0f + i * 5.0f;  // 2, 7, 12, 17, ...
        glPointSize(pointSize);
        
        // 设置颜色(渐变色:从蓝到红)
        float r = (float)i / (numPoints - 1);
        float b = 1.0f - r;
        glColor3f(r, 0.5f, b);
        
        // 绘制点
        glBegin(GL_POINTS);
        glVertex2f(x, y);
        glEnd();
    }
    
    // 刷新缓冲区
    glFlush();
}

void init() {
    // 设置背景色(白色)
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    
    // 启用抗锯齿(让点更圆滑)
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

int main(int argc, char** argv) {
    // 初始化 GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("不同大小的点");
    
    // 初始化 OpenGL 设置
    init();
    
    // 注册回调函数
    glutDisplayFunc(display);
    
    // 进入主循环
    glutMainLoop();
    
    return 0;
}
