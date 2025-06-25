#include <cmath>
#include <iomanip>
#include <iostream>
using namespace std;

class Shape {
public:
  virtual double area() = 0;
};

class Circle : public Shape {
public:
    Circle(double cinr) {r= cinr;}
    virtual double area(){
        return M_PI*r*r;
    }
    double r;
  // TODO: Implement this class
};

class Rectangle : public Shape {
public:
    Rectangle(double cinlength, double cinwidth){
        length= cinlength;
        width= cinwidth;
    }
    virtual double area(){
        return length*width;
    }
    double length;
    double width;
  // TODO: Implement this class
};

class Triangle : public Shape {
public:
    Triangle(double l1, double l2, double l3){
        line1= l1;
        line2= l2;
        line3= l3;
    }
    virtual double area(){
        double p= (line1+line2+line3)/2;
        return sqrt(p*(p-line1)*(p-line2)*(p-line3));
    }
    double line1;
    double line2;
    double line3;

  // TODO: Implement this class
};

int main() {
    int n;
    cin >> n;
    Shape *shapes[n];

    // TODO: Read input and create objects
    for(int i= 0; i<n; i++){
        string name;
        cin>>name;
        if(name== "Circle"){
            double r;
            cin>>r;
            shapes[i]= new Circle(r);
        }else if(name== "Rectangle"){
            double length, width;
            cin>>length>>width;
            shapes[i]= new Rectangle(length, width);
        }else if(name== "Triangle"){
            double l1, l2, l3;
            cin>>l1>>l2>>l3;
            shapes[i]= new Triangle(l1, l2, l3);
        }
    }
    cout << fixed << setprecision(2);
    for (int i = 0; i < n; ++i) {
        cout << shapes[i]->area() << endl;
    }

    return 0;
}
