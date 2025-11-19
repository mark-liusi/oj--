#include <GL/glut.h>  
void  init(void)
{  glClearColor (0.0,0.0,0.0,0.0);    //设置背景色
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity( ); //初始化变换矩阵
gluOrtho2D(0.0,300.0,0.0,60.0);//用来设定屏幕窗口所需要的变换（裁剪） 
}
void  display(void)
{   int i=0;
	glClear(GL_COLOR_BUFFER_BIT);    //清屏
    glColor3f(1.0,1.0,1.0);              //白色
	glPointSize(1.0f);                         //点的大小为一个像素
	for(i=0; i<10; i++)                  //画第一组点
	{	glBegin(GL_POINTS);
		    glVertex2f(20.0f+i*30.0f,50.0f);
		glEnd( );
	}
	glColor3f(0.0f,1.0f,0.0f);              //绿色
	glPointSize(3.0f);                      //点的大小为三个像素
	for(i=0; i<10; i++)                        //画第二组点
	{	glBegin(GL_POINTS);
		    glVertex2f(20.0f+i*30.0f,30.0f);
		glEnd( );
	}
	glColor3f(0.0f,0.0f,1.0f);              //蓝色
	glPointSize(5.0f);                         //点的大小为五个像素
	for(i=0; i<10; i++)                        //画第三组点
	{	glBegin(GL_POINTS);
		     glVertex2f(20.0f+i*30.0f,10.0f);
		glEnd( );
	}
	glFlush( );
}
int  main( int argc, char** argv )
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_SINGLE|GLUT_RGB);
	glutInitWindowSize(300,60);
	glutInitWindowPosition(100,100);
	glutCreateWindow("hello");
	init( );
	glutDisplayFunc(display);
	glutMainLoop();
	return 0; 
}