#include <GL/glut.h>

void init(void) {
    glClearColor(1.0, 1.0, 1.0, 0.0);  // 白色背景
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, 800.0, 0.0, 600.0);  // 设置坐标系统
}

void display(void) {
    glClear(GL_COLOR_BUFFER_BIT);
    
    // 1. 绘制紫色矩形框 (左上角)
    glColor3f(0.6f, 0.3f, 0.8f);  // 紫色
    glLineWidth(3.0f);
    glBegin(GL_LINE_LOOP);
        glVertex2f(100.0f, 400.0f);  // 左下
        glVertex2f(350.0f, 400.0f);  // 右下
        glVertex2f(350.0f, 520.0f);  // 右上
        glVertex2f(100.0f, 520.0f);  // 左上
    glEnd();
    
    // 2. 绘制三个蓝色点 (右侧) - 一个在上，两个在下
    glColor3f(0.0f, 0.0f, 1.0f);  // 蓝色
    glPointSize(15.0f);
    glBegin(GL_POINTS);
        glVertex2f(600.0f, 480.0f);  // 上面的点
        glVertex2f(520.0f, 380.0f);  // 左下的点
        glVertex2f(680.0f, 380.0f);  // 右下的点
    glEnd();
    
    // 3. 绘制绿色五边形箭头 (左下角)
    glColor3f(0.3f, 0.9f, 0.2f);  // 绿色
    glBegin(GL_POLYGON);
        glVertex2f(100.0f, 200.0f);  // 左上
        glVertex2f(250.0f, 200.0f);  // 中上
        glVertex2f(400.0f, 150.0f);  // 右尖端
        glVertex2f(250.0f, 100.0f);  // 中下
        glVertex2f(100.0f, 100.0f);  // 左下
    glEnd();
    
    glFlush();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(800, 600);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("liusi 20236456");  // 窗口标题
    init();
    glutDisplayFunc(display);
    glutMainLoop();
    return 0;
}
