#include <cuda/std/tuple>
#include <cuda/std/cmath>
#include <iostream>

struct Point2D { float x, y;};
struct Point3D { float x, y, z;};


class Normal{
public:
    int64_t nx, ny, nz;
    float nxf, nyf, nzf;
    int64_t constant;
    double constantf;
};

class Triangles2D{
public :
    Point2D p1i, p2i, p3i;  // position
    Point3D p1_3d, p2_3d, p3_3d;// position
    float area;
    int64_t square_x, square_y, square_width , square_height;
    bool only_face;
    Normal normal;

};

__device__ float sign(Point2D p1, Point2D p2, Point2D p3){
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

extern "C" __global__ void gradiant(const Triangles2D* __restrict__ triangles, unsigned int* __restrict__ Color, unsigned int triangles_len, int width, int height, int y_g, int x_g, int grid_size){
    int index = threadIdx.x;
    int x_i = ((index%grid_size)) + grid_size*x_g;
    int y_i = ((index/grid_size)) + grid_size*y_g;
    int idx = x_i + y_i * width;
    if (x_i < width && y_i < height){
        float x = (float)(x_i - width/2);
        float y = (float)(y_i - height/2);
        Color[idx] = 0xFF00FF;
        for (unsigned int i = 0; i < triangles_len; i++){
            const Triangles2D& tri = triangles[i];
	        float d1 = sign(Point2D{ x, y } , tri.p1i, tri.p2i);
            float d2 = sign(Point2D{ x, y } , tri.p2i, tri.p3i);
            float d3 = sign(Point2D{ x, y } , tri.p3i, tri.p1i);
            if ((cuda::std::signbit(d1) == cuda::std::signbit(d2)) && (cuda::std::signbit(d2) == cuda::std::signbit(d3))){
                Color[idx] = 0xFFFFFF;
            }
            //}
        }
    }
}
