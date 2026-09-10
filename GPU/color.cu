#include <cuda/std/tuple>
#include <cuda/std/cmath>
#include <iostream>

struct Point2D { float x, y;};
struct Point3D { float x, y, z;};

__device__ float vec_size(const Point3D& p){
    return cuda::std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

class Normal{
public:
    float nxf, nyf, nzf;
    int64_t constant;
    float constantf;

    __device__ float dot_product(Point3D& p) const {
        return (nxf*p.x) + (nyf*p.y) + (nzf*p.z);
    }

    __device__ float size() const {
        return vec_size(Point3D{nxf,nyf,nzf});
    }
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

__device__ float sign(const Point2D& p1, const Point2D& p2, const Point2D& p3){
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

__device__ float dot(Point3D& p1, Point3D& p2){
    return (p1.x*p2.x) + (p1.y*p2.y) + (p1.z * p2.z);
}

__device__ float intersection_t(const Normal& normal, Point3D& droite, Point3D& point){
    return -(normal.dot_product(point) + normal.constantf)/normal.dot_product(droite);
}
__device__ Point3D intersection_xyz(const Normal& normal, Point3D& droite, Point3D& point){
    float t = intersection_t(normal, droite, point);
    float x = point.x + droite.x * t;
    float y = point.y + droite.y * t;
    float z = point.z + droite.z * t;
    return Point3D{ x, y, z };
}

__device__ float intersection_z(const Normal& normal, Point3D& droite, Point3D& point){
    float t = intersection_t(normal, droite, point);
    float z = point.z + droite.z * t;
    return z;
}

__device__ Point3D sub(Point3D& p1, Point3D& p2){
    return Point3D{p1.x - p2.x, p1.y - p2.y, p1.z - p2.z};
}

__device__ Point3D cross(Point3D& a, Point3D& b){
    return Point3D{ a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}

__device__ float clamp(float value, float min, float max){
    return fminf(fmaxf(value,min),max);
}

const float CAMERA_FOCAL_DIST = 350.0;

extern "C" __global__ void color(const Triangles2D* __restrict__ triangles,unsigned int* __restrict__ Color, int x_c, int y_c, int grid_size, int width, int height, int triangles_len){
    int x_t = threadIdx.x + blockDim.x * blockIdx.x;
    int y_t = threadIdx.y + blockDim.y * blockIdx.y;

    int fac_x = width/grid_size;
    int fac_y = height/grid_size;
    int x_i = x_t + x_c * fac_x;
    int y_i = y_t + y_c * fac_y;
    if (x_i < (x_c + 1)*fac_x && y_i < (y_c + 1)*fac_y){
        int index = x_i + y_i*width;
        if (index < width*height){
            //Color[index] = ((grid_size*9876567239 + x_c*9876 +y_c*98765678)%(0xFFFFFF));
            Color[index] = 0;
            float x = (float)(x_i - width/2);
            float y = (float)(y_i - height/2);
            float mini_z = -1.0;
            int tri_index = 0;
            Point3D val = Point3D{x,y,CAMERA_FOCAL_DIST};
            for (unsigned int i = 0; i < triangles_len; i++){
                const Triangles2D& tri = triangles[i];
                float d1 = sign(Point2D{ x, y } , tri.p1i, tri.p2i);
                float d2 = sign(Point2D{ x, y } , tri.p2i, tri.p3i);
                float d3 = sign(Point2D{ x, y } , tri.p3i, tri.p1i);
                if ((cuda::std::signbit(d1) == cuda::std::signbit(d2)) && (cuda::std::signbit(d2) == cuda::std::signbit(d3))){
                    float z = intersection_z(tri.normal, val, val);
                    if (mini_z == -1 || (z < mini_z && z > CAMERA_FOCAL_DIST)){
                        tri_index = i;
                        mini_z = z;
                    }
                }
            }
            if (mini_z != -1){
                const Triangles2D& tri = triangles[tri_index];
                Point3D p = intersection_xyz(tri.normal,val,val);

                //float u = tri.normal.dot_product(cross(sub(tri.p2_3d, p),sub(tri.p3_3d,p))) / tri.area;
                //float v = tri.normal.dot_product(cross(sub(tri.p3_3d, p),sub(tri.p1_3d,p))) / tri.area;
                //float w = 1.0 - u - v;


                //float x = (tri_3d.uv1.0 * u)+(tri_3d.uv2.0 * v)+(tri_3d.uv3.0 * w);
                //float y = (tri_3d.uv1.1 * u)+(tri_3d.uv2.1 * v)+(tri_3d.uv3.1 * w);
                float r = 0.0;
                float g = 0.0;
                float b = 0.0;
                float r_c = 255.0;
                float g_c = 255.0;
                float b_c = 255.0;
                Point3D vec = Point3D{ -p.x , -p.y, -p.z};
                float dt = -tri.normal.dot_product(vec);
                if (dt != 0.0){
                    float dist = vec_size(vec);
                    float sz = dist * tri.normal.size();
                    float angle = dt / sz;
                    if (angle > 0.0){
                        float light = 255.0 * angle * 10.0;
                        float fac = 20.0/dist;
                        fac = fac*fac;
                        float dim_r = clamp(r_c * fac,0.0,1.0);
                        float dim_g = clamp(g_c * fac,0.0,1.0);
                        float dim_b = clamp(b_c * fac,0.0,1.0);
                        r = clamp(r + (light*dim_r),0,255);
                        g = clamp(g + (light*dim_g),0,255);
                        b = clamp(b + (light*dim_b),0,255);
                    }
                }
                unsigned int color = (((unsigned int)r) <<16) + (((unsigned int) g) << 8) + ((unsigned int)b);
                Color[index] = color;
            }
        }
    }
}