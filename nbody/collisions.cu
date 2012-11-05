

#include "Vector.hpp"
#include "GLee.h"
#include "glut.h"
#include <math.h>

// For Vector3
using namespace Kamm;


// OBJECTS

class Sphere
{
public:
    explicit Sphere(float r, float m) :
    radius(r),
        mass(m),
        pos(),
        vel(),
        acc()
    {
        memset(color, 0, sizeof(color));
    }

    virtual ~Sphere()
    {

    }

    void draw() const
    {
        glTranslatef(pos[0], pos[1], pos[2]);
        glColor4fv( color );
        glutSolidSphere(radius, 16, 16);
    }


public:
    float       radius;
    float       mass;
    Vector3     pos;
    Vector3     vel;
    Vector3     acc;
    float       color[4];

};

// GLOBALS / CONSTANTS
const size_t    objects = 3;
Sphere          *spheres[ objects ];
bool            advanced = false;


// PHYSICS

bool simpleSphereSphere(Sphere *a, Sphere *b)
{
    // This is really simple. We take the sum of the radii and compares
    // it to the distance between the spheres. If distance is less than the
    // sum of the radii, they intersect and they collides.

    Vector3 distance = a->pos - b->pos;

    float length = distance.length();

    // Sum of the radiuses
    float sumradius = a->radius + b->radius;

    if (length <= sumradius)
    {
        return true;
    }

    return false;
}

bool advancedSphereSphere(Sphere *a, Sphere *b, float &t)
{
    Vector3 s = a->pos - b->pos; // vector between the centers of each sphere
    Vector3 v = a->vel - b->vel; // relative velocity between spheres
    float r = a->radius + b->radius;

    float c1 = s.dot(s) - r*r; // if negative, they overlap
    if (c1 < 0.0) // if true, they already overlap
    {
        // This is bad ... we need to correct this by moving them a tiny fraction from each other
        //a->pos +=
        t = .0;
        return true;
    }

    float a1 = v.dot(v);
    if (a1 < 0.00001f)
        return false; // does not move towards each other

    float b1 = v.dot(s);
    if (b1 >= 0.0)
        return false; // does not move towards each other

    float d1 = b1*b1 - a1*c1;
    if (d1 < 0.0)
        return false; // no real roots ... no collision

    t = (-b1 - sqrtf(d1)) / a1;

    return true;
}

void sphereCollisionResponse(Sphere *a, Sphere *b)
{
    Vector3 U1x,U1y,U2x,U2y,V1x,V1y,V2x,V2y;


    float m1, m2, x1, x2;
    Vector3 v1temp, v1, v2, v1x, v2x, v1y, v2y, x(a->pos - b->pos);

    x.normalize();
    v1 = a->vel;
    x1 = x.dot(v1);
    v1x = x * x1;
    v1y = v1 - v1x;
    m1 = a->mass;

    x = x*-1;
    v2 = b->vel;
    x2 = x.dot(v2);
    v2x = x * x2;
    v2y = v2 - v2x;
    m2 = b->mass;

    a->vel = Vector3( v1x*(m1-m2)/(m1+m2) + v2x*(2*m2)/(m1+m2) + v1y );
    b->vel = Vector3( v1x*(2*m1)/(m1+m2) + v2x*(m2-m1)/(m1+m2) + v2y );
}

void doGravity()
{
    Vector3 center(0,0,0);

    for (size_t i=0; i<objects; ++i)
    {
        Sphere *a = spheres[i];

        // Attract to center 0,0,0
        Vector3 direction = (center - a->pos) / a->pos.length();

        // Add to acc vector
        a->acc = direction / 1.0f;
    }
}

void doPhysics(float dt, bool step = false)
{
    // Let the games begin

    // 1. Acceleration
    doGravity();

    // 2. Speeds

    // Distribute gravity into speed
    for (size_t i=0; i<objects; ++i)
    {
        Sphere *a = spheres[i];

        // Multiply with 0.98 to damp the motions
        a->vel += a->acc * dt * 0.98f;
    }

    for (size_t i=0; i<objects; ++i)
    {
        for (size_t j=i+1; j<objects; ++j)
        {
            Sphere *a = spheres[i];
            Sphere *b = spheres[j];

            if (advanced)
            {
                float timetocollision;
                if (advancedSphereSphere(a, b, timetocollision) || step)
                {
                    if (timetocollision < dt && !step)
                    {
                        // Move simulation forward by a bit and do physics
                        doPhysics(timetocollision, true);
                    }

                    if (step)
                    {
                        sphereCollisionResponse(a,b);
                    }
                }
            }
            else
            {
                if (simpleSphereSphere(a,b))
                {
                    sphereCollisionResponse(a,b);
                }
            }
        }
    }

    // 3. Update positions
    for (size_t i=0; i<objects; ++i)
    {
        Sphere *a = spheres[i];

        a->pos += a->vel * dt;
    }
}


// GLUT / OPENGL

void changeSize(int w, int h)
{
    // Prevent a divide by zero, when window is too short
    // (you cant make a window of zero width).
    if(h == 0)
        h = 1;

    //w = w1;
    //h = h1;
    GLdouble ratio = 1.0f * w / h;
    // Reset the coordinate system before modifying
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // Set the viewport to be the entire window
    glViewport(0, 0, w, h);

    // Set the clipping volume
    gluPerspective(45,ratio,0.1,1000);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

}

void drawCoords()
{
    ///*
    glPushMatrix();
    glBegin(GL_LINES);
    // X in red...
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(1, 0, 0);

    // Y in green
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 1, 0);

    // Z in blue
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, 1);

    glColor3f(1.0f, 1.0f, 1.0f);

    // floor
    for (int xx=-50; xx<50; xx++)
    {
        glVertex3f(xx, 0, -50);
        glVertex3f(xx, 0, 50);
        glVertex3f(50, 0, xx);
        glVertex3f(-50, 0, xx);
    }
    glEnd();
    glPopMatrix();
}


void renderScene()
{
    doPhysics(1.0 / 60.0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    gluLookAt(20.0f, 2.0f, 0.0f,
//      0.0f, 0.0f, 0.0f,
        spheres[0]->pos[0],
        spheres[0]->pos[1],
        spheres[0]->pos[2],
        0.0f, 1.0f, 0.0f);


    drawCoords();

    for (size_t i=0; i<objects; ++i)
    {
        glPushMatrix();
        spheres[i]->draw();
        glPopMatrix();
    }


    Sleep(10);
    glutSwapBuffers();

}




// MAIN

int main()
{
    // Init Spheres
    spheres[0] = new Sphere(1,1);
    spheres[1] = new Sphere(1,2);
    spheres[2] = new Sphere(1.5,4);

    spheres[0]->pos = Vector3(0,3,-0.1);
    spheres[1]->pos = Vector3(0,-3,0.1);
    spheres[2]->pos = Vector3(0,1,5);

    spheres[0]->vel = Vector3(0,-0.5,0);
    spheres[1]->vel = Vector3(0,0.5,0);
    spheres[2]->vel = Vector3(0,0,0);

    // Colors
    spheres[0]->color[0] = 1.0f;
    spheres[1]->color[1] = 1.0f;
    spheres[2]->color[2] = 1.0f;

    // Init GLUT

    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(640,480);
    glutCreateWindow("Simple sphere-sphere collision!");

    glEnable(GL_DEPTH_TEST);
    glutIgnoreKeyRepeat(1);

    glutReshapeFunc(changeSize);
    glutDisplayFunc(renderScene);
    glutIdleFunc(renderScene);

    glutMainLoop();

    return 0;
}

