mod GPU;
mod parse_config;

use std::ffi::c_void;
use std::{ptr, time};
use minifb::{Key, MouseMode, Window, WindowOptions};

use std::time::Instant;
use image::RgbaImage;
use minifb::Key::{H, W};
use rayon::prelude::*;
use GPU::Gpu;
use crate::GPU::{dim3, CUdeviceptr};
use parse_config::generate_config;


const WIDTH: usize = 1920;
const WIDTH2I : i64 = (WIDTH as i64) /2;
const HEIGHT: usize = 1080;
const HEIGHT2I : i64 = (HEIGHT as i64) /2;

#[allow(dead_code)]
fn simple_rng(min : usize,max : usize) -> i64{
    static mut SEED: usize = 100;
    unsafe {
        SEED = SEED.wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (SEED % (max-min) + min)  as i64
    }
}


const FPS_BUFFER_SIZE: usize = 500;
const FPS_BUFFER_SIZE_128: u128 = FPS_BUFFER_SIZE as u128;
const FPS : usize = 60;


#[derive(Clone)]
struct Color{
    img : Option<RgbaImage>,
    width : f32,
    height : f32,
    color : (u32,u32,u32,u32),
}

impl Color {
    pub const fn new_from_u32(color : u32) ->  Color{
        Color{
            img: None,
            width: 0.0,
            height: 0.0,
            color : (color >> 16,(color & 0x00FF00) >> 8,color & 0x0000FF,255),
        }
    }

    pub fn new_from_img( img : RgbaImage ) -> Color {
        let width  = img.width() as f32;
        let height = img.height() as f32;
        Color{
            img: Some(img),
            width,
            height,
            color: (0,0,0,255),
        }
    }

    pub fn get_barycentric(&self, x : f32, y : f32) -> (u32,u32,u32,u32){
        if let Some(img) = &self.img {
            let xi = ((((x * self.width) % self.width) + self.width ) % self.width) as u32;
            let yi = ((((y * self.height) % self.height) + self.height) % self.height) as u32;
            let pix = img.get_pixel(xi, yi).0;
            (pix[0] as u32, pix[1] as u32, pix[2] as u32, pix[3] as u32)
        }else{
            self.color
        }

    }
}

#[repr(C)]
#[derive(Debug,Clone)]
struct  Point2D{
    pub x : f32,
    pub y : f32,
}

#[repr(C)]
#[derive(Debug,Clone)]
struct Point3D{
    pub x : f32,
    pub y : f32,
    pub z : f32,
}

#[derive(Debug)]
struct Camera{
    focal_dist : i64,
    focal_distf : f32,
    pos : (i64,i64,i64),
    posf : (f32,f32,f32),
    rot : (f32,f32,f32),
    rot_cos_sin_1_2 : ((f32,f32),(f32,f32),(f32,f32)),
    normal: Normal,
    sun : (f32,f32,f32),
    lights : Vec<(f32,f32,f32)>,
    old_mouse_pos : (f32,f32),
}

impl Camera{
    pub const fn new(focal_point_distance : i64,sun : (f32,f32,f32), lights : Vec<(f32,f32,f32)>) -> Camera{
        Camera{
            //focal_point: (0, 0, -focal_point_distance),
            focal_dist: focal_point_distance,
            focal_distf: focal_point_distance as f32,
            pos : (0, 0, 0),
            posf: (0.0, 0.0, 0.0),
            rot : (0f32, 0f32, 0f32),
            rot_cos_sin_1_2: ((0.0, 0.0),(0.0, 0.0), (0.0, 0.0)),
            normal: Normal::new_from_equation((0, 0, 1), (0, 0, focal_point_distance)),
            sun,
            lights,
            old_mouse_pos: (0.0, 0.0),
        }
    }

    pub fn move_camera(&mut self, direction : (i64,i64,i64)){
        let x = ((direction.0 as f32)*(-self.rot.1).cos() - (direction.2 as f32)*(-self.rot.1).sin()) as i64;
        let z = ((direction.0 as f32)*(-self.rot.1).sin() + (direction.2 as f32)*(-self.rot.1).cos()) as i64;
        self.pos.0 +=  x;
        self.pos.1 += direction.1;
        self.pos.2 += z;
        self.posf.0 = self.pos.0 as f32;
        self.posf.1 = self.pos.1 as f32;
        self.posf.2 = self.pos.2 as f32;
    }

    #[inline(always)]
    fn point_transformation(&self ,point : (f32,f32,f32)) -> (f32,f32,f32) {
        let point_c = (point.0 - self.posf.0, point.1 - self.posf.1, point.2 - self.posf.2);
        let nx     = point_c.0 * self.rot_cos_sin_1_2.1.0 - point_c.2 * self.rot_cos_sin_1_2.1.1;
        let nz_tmp = point_c.0 * self.rot_cos_sin_1_2.1.1 + point_c.2 * self.rot_cos_sin_1_2.1.0;
        let nz     = point_c.1 * self.rot_cos_sin_1_2.2.1 + nz_tmp * self.rot_cos_sin_1_2.2.0;
        let ny     = point_c.1 * self.rot_cos_sin_1_2.2.0 - nz_tmp * self.rot_cos_sin_1_2.2.1;
        (nx, ny, nz)
    }
    pub fn get_lights(&self) -> Vec<(f32,f32,f32)>{
        let mut lights : Vec<(f32,f32,f32)> = Vec::new();
        for light in self.lights.iter(){
            lights.push(self.point_transformation(*light))
        };
        lights
    }

    pub fn move_cam(&mut self, window: &Window) {
        let mut speed = 10;
        if let Some(pos) = window.get_mouse_pos(MouseMode::Discard) {
            if self.old_mouse_pos == (0f32, 0f32) || self.old_mouse_pos == pos {
                self.old_mouse_pos = pos;
            } else {
                let diff_x = (pos.0 - self.old_mouse_pos.0) / 5000f32;
                let diff_y = (pos.1 - self.old_mouse_pos.1) / 5000f32;
                self.rot.1 += diff_x.to_degrees();
                self.rot.2 += diff_y.to_degrees();
                self.rot_cos_sin_1_2 = ((self.rot.0.cos(),self.rot.0.sin()),
                                        (self.rot.1.cos(),self.rot.1.sin()),
                                        (self.rot.2.cos(),self.rot.2.sin()));
                self.old_mouse_pos = (pos.0, pos.1);
            }
        } else {
            self.old_mouse_pos = (0f32, 0f32)
        }
        if window.is_key_down(Key::LeftShift) {
            speed = 100;
        }
        if window.is_key_down(Key::X) {
            self.rot.1 -= 1f32.to_radians();
        }
        if window.is_key_down(Key::D) {
            self.move_camera((speed, 0, 0));
        }
        if window.is_key_down(Key::Q) {
            self.move_camera((-speed, 0, 0));
        }
        if window.is_key_down(Key::Z) {
            self.move_camera((0, 0, speed));
        }
        if window.is_key_down(Key::S) {
            self.move_camera((0, 0, -speed));
        }
        if window.is_key_down(Key::A){
            self.move_camera((0, speed, 0));
        }
        if window.is_key_down(Key::E){
            self.move_camera((0, - speed,0));
        }
    }
}


#[repr(C)]
#[derive(Clone,Debug)]
struct Normal{
    nxf : f32,
    nyf : f32,
    nzf : f32,
    constant : i64,
    constantf : f32
}

impl Normal{
    pub const fn new(p1 : (f32,f32,f32), p2 : (f32,f32,f32), p3 : (f32,f32,f32)) -> Normal{
        let a = (p2.0 - p1.0, p2.1 - p1.1, p2.2 - p1.2);
        let b = (p3.0 - p1.0, p3.1 - p1.1, p3.2 - p1.2);
        let nx = a.1* b.2 - a.2* b.1;
        let ny = a.2* b.0 - a.0* b.2;
        let nz = a.0* b.1 - a.1* b.0;
        let constant = -1f32 * (nx*p1.0 + ny*p1.1 + nz*p1.2);
        Normal {
            nxf : nx,
            nyf : ny,
            nzf : nz,
            constantf: constant,
            constant: constant as i64,
        }
    }

    pub const fn new_from_equation_f32(eq : (f32,f32,f32), pt : (f32,f32,f32)) -> Normal{
        Normal{
            nxf: eq.0,
            nyf: eq.1,
            nzf: eq.2,
            constant: -1 * (eq.0*pt.0 + eq.1*pt.1 + eq.2*pt.2) as i64,
            constantf: -1f32 * (eq.0*pt.0 + eq.1*pt.1 + eq.2*pt.2),
        }
    }

    pub const fn new_from_equation(eq : (i64,i64,i64), pt : (i64,i64,i64)) -> Normal{
        Normal{
            nxf: eq.0 as f32,
            nyf: eq.1 as f32,
            nzf: eq.2 as f32,
            constant: -1 * (eq.0*pt.0 + eq.1*pt.1 + eq.2*pt.2),
            constantf: (-1 * (eq.0*pt.0 + eq.1*pt.1 + eq.2*pt.2)) as f32,
        }
    }

    pub const fn dot_product(&self, vec : (i64,i64,i64)) -> i64 {
        (self.nxf as i64)*vec.0 + (self.nyf as i64)*vec.1 + (self.nzf as i64)*vec.2
    }

    pub const fn dot_product_f32(&self, vec:(f32,f32,f32)) -> f32{
        vec.0 * self.nxf +vec.1 * self.nyf +vec.2 * self.nzf
    }

    #[inline(always)]
    pub const fn inverse(&mut self){
        self.nxf *= -1.0;
        self.nyf *= -1.0;
        self.nzf *= -1.0;
        self.constant *= -1;
        self.constantf *= -1f32;
    }

    #[inline(always)]
    pub const fn to_vec_f32(&self) -> (f32,f32,f32){
        (self.nxf,self.nyf,self.nzf)
    }
}

struct Triangle3D{
    p1 : (f32,f32,f32),
    p2 : (f32,f32,f32),
    p3 : (f32,f32,f32),
    color : &'static Color,
    inverse_normal : bool,
    normal : Normal,
    only_face : bool,
    uv1 : (f32,f32),
    uv2 : (f32,f32),
    uv3 : (f32,f32),
    area : f32
}

impl Triangle3D{
    pub fn new(p1 : (f32,f32,f32), p2 : (f32,f32,f32), p3 : (f32,f32,f32), color : &'static Color, only_face : bool,inverse_normal : bool) -> Triangle3D{
        let mut normal = Normal::new(p1,p2,p3);
        if inverse_normal{
            normal.inverse();
        }
        let area = dot(normal.to_vec_f32(), cross(sub(p2, p1), sub(p3, p1)));
        Triangle3D{p1 ,p2 ,p3 ,color , inverse_normal, normal, only_face, uv1: (0.0, 0.0), uv2: (0.0, 1.0), uv3: (1.0, 1.0), area}
    }

    pub fn new_stl(p1:(f32,f32,f32), p2: (f32,f32,f32), p3 : (f32,f32,f32),normal: Normal, color : &'static Color)-> Triangle3D{
        let area = dot(normal.to_vec_f32(), cross(sub(p2, p1), sub(p3, p1)));
        Triangle3D{
            p1,
            p2,
            p3,
            color,
            inverse_normal: false,
            normal,
            only_face: false,
            uv1: (0.0, 0.0),
            uv2: (0.0, 1.0),
            uv3: (1.0, 1.0),
            area,
        }
    }

    pub fn new_obj(p1:(f32,f32,f32), p2: (f32,f32,f32), p3 : (f32,f32,f32), uv1 : (f32,f32), uv2 : (f32,f32), uv3 : (f32,f32), color : &'static Color)-> Triangle3D{
        let mut normal = Normal::new(p1,p2,p3);
        normal.inverse();
        let area = dot(normal.to_vec_f32(), cross(sub(p2, p1), sub(p3, p1)));
        Triangle3D{
            p1,
            p2,
            p3,
            color,
            inverse_normal: false,
            normal,
            only_face: false,
            uv1,
            uv2,
            uv3,
            area,
        }
    }

    pub fn to_2d(&self, camera: &Camera) ->  Vec<Triangle2D>{
        let p1_3d = camera.point_transformation(self.p1);
        let p2_3d = camera.point_transformation(self.p2);
        let p3_3d = camera.point_transformation(self.p3);

        if p1_3d.2 < camera.focal_distf && p2_3d.2 < camera.focal_distf && p3_3d.2 < camera.focal_distf {
            vec![]
        }else {
            let mut triangles = Vec::with_capacity(2);
            let mut normal = Normal::new(p1_3d,p2_3d,p3_3d);
            if self.inverse_normal{
                normal.inverse()
            }
            let p1 : (f32,f32);
            let p2 : (f32,f32);
            let p3 : (f32,f32);
            if p1_3d.2 < camera.focal_distf {
                if p2_3d.2 < camera.focal_distf {
                    p1 = plane_clipping(p1_3d,p3_3d,&camera.normal);
                    p2 = plane_clipping(p2_3d,p3_3d,&camera.normal);
                    p3 = projection(p3_3d,&camera.normal);
                }else if p3_3d.2 < camera.focal_distf {
                    p1 = plane_clipping(p1_3d,p2_3d,&camera.normal);
                    p2 = projection(p2_3d,&camera.normal);
                    p3 = plane_clipping(p3_3d,p2_3d,&camera.normal);
                }else{
                    let pr2 = plane_clipping(p2_3d,p1_3d,&camera.normal);
                    let pr3 = plane_clipping(p3_3d,p1_3d,&camera.normal);
                    p2 = projection(p2_3d,&camera.normal);
                    p3 = projection(p3_3d,&camera.normal);
                    p1 = pr2;
                    triangles.push(Triangle2D::new(pr2,pr3,p3,p1_3d,p2_3d,p3_3d,normal.clone(),self.only_face));
                }
            }else if p2_3d.2 < camera.focal_distf {
                if p3_3d.2 < camera.focal_distf{
                    p1 = projection(p1_3d,&camera.normal);
                    p2 = plane_clipping(p2_3d,p1_3d,&camera.normal);
                    p3 = plane_clipping(p3_3d,p1_3d,&camera.normal);
                }else{
                    let pr1 = plane_clipping(p1_3d,p2_3d,&camera.normal);
                    let pr3 = plane_clipping(p3_3d,p2_3d,&camera.normal);
                    p1 = projection(p1_3d,&camera.normal);
                    p3 = projection(p3_3d,&camera.normal);
                    p2 = pr1;
                    triangles.push(Triangle2D::new(pr1,pr3,p3,p1_3d,p2_3d,p3_3d,normal.clone(),self.only_face));
                }
            }else if p3_3d.2 < camera.focal_distf {
                let pr1 = plane_clipping(p1_3d,p3_3d,&camera.normal);
                let pr2 = plane_clipping(p2_3d,p3_3d,&camera.normal);
                p1 = projection(p1_3d,&camera.normal);
                p2 = projection(p2_3d,&camera.normal);
                p3 = pr1;
                triangles.push(Triangle2D::new(pr1,pr2,p2,p1_3d,p2_3d,p3_3d,normal.clone(),self.only_face));
            }else{
                p1 = projection(p1_3d,&camera.normal);
                p2 = projection(p2_3d,&camera.normal);
                p3 = projection(p3_3d,&camera.normal);
            }
            triangles.push(Triangle2D::new(p1, p2, p3,p1_3d,p2_3d,p3_3d, normal,self.only_face));
            triangles
        }
    }
}
#[inline(always)]
fn plane_clipping(p1 : (f32,f32,f32), p2 : (f32,f32,f32), normal: &Normal) -> (f32,f32){
    let vec = (p2.0 - p1.0, p2.1 - p1.1, p2.2 - p1.2);
    intersection_xy(normal, vec, p1)
}



#[inline(always)]
const fn projection(point : (f32,f32,f32), normal: &Normal) -> (f32,f32){
    if is_secant(normal,(point.0 as i64,point.1 as i64, point.2 as i64)) {
        intersection_xy(normal,point,point)
    }else{
        (0f32,0f32)
    }
}

#[repr(C)]
#[derive(Debug,Clone)]
struct Triangle2D{
    p1i : Point2D, // position
    p2i : Point2D, // position
    p3i : Point2D, // position
    p1_3d : Point3D, // position
    p2_3d : Point3D, // position
    p3_3d : Point3D, // position
    area : f32,
    square_x : i64,
    square_y : i64,
    square_width : i64,
    square_height : i64,
    only_face : bool,
    normal: Normal,
}

impl Triangle2D{
    pub const fn new(p1i : (f32,f32), p2i : (f32,f32), p3i : (f32,f32), p1_3d : (f32,f32,f32), p2_3d: (f32,f32,f32), p3_3d: (f32,f32,f32), normal: Normal, only_face : bool) -> Triangle2D{
        let p1 = (p1i.0 as i64,p1i.1 as i64);
        let p2 = (p2i.0 as i64,p2i.1 as i64);
        let p3 = (p3i.0 as i64,p3i.1 as i64);
        let area = dot(normal.to_vec_f32(), cross(sub(p2_3d, p1_3d), sub(p3_3d, p1_3d)));
        Triangle2D{
            p1i : Point2D{x : p1i.0, y: p1i.1},
            p2i : Point2D{x : p2i.0, y: p2i.1},
            p3i : Point2D{x : p3i.0, y: p3i.1},
            p1_3d : Point3D{ x : p1_3d.0, y : p1_3d.1 , z: p1_3d.2},
            p2_3d : Point3D{ x : p2_3d.0, y : p2_3d.1 , z: p2_3d.2},
            p3_3d : Point3D{ x : p3_3d.0, y : p3_3d.1 , z: p3_3d.2},
            area,
            square_x: min3(p1.0, p2.0, p3.0),
            square_y: min3(p1.1, p2.1, p3.1),
            square_width: max3(p1.0, p2.0, p3.0),
            square_height: max3(p1.1, p2.1, p3.1),
            only_face,
            normal,
        }
    }
}

#[inline(always)]
const fn sign(p1 : (i64,i64), p2 : (f32,f32), p3 : (f32,f32)) -> f32{
    (p1.0 as f32 - p3.0) * (p2.1 - p3.1) - (p2.0 - p3.0) * (p1.1 as f32 - p3.1)
}

#[inline(always)]
const fn min3(v1 : i64, v2 : i64, v3 : i64) -> i64{
    if v1 < v2{
        if v1 < v3 {v1} else {v3}
    }else{
        if v3 < v2 {v3} else {v2}
    }
}

#[inline(always)]
const fn max3(v1 : i64, v2 : i64, v3 : i64) -> i64{
    if v1 > v2 {
        if v1 > v3 {v1} else {v3}
    }else{
        if v3 > v2 {v3} else {v2}
    }
}

#[inline(always)]
const fn is_secant(normal: &Normal, droite : (i64,i64,i64)) -> bool{
    normal.dot_product(droite) != 0
}

#[inline(always)]
const fn intersection_t(normal : &Normal, droite: (f32,f32,f32), point : (f32,f32,f32)) -> f32{
    -(normal.dot_product_f32(point)  + normal.constantf)/ normal.dot_product_f32(droite)
}
#[inline(always)]
const fn intersection_z(value : i64, prod : i64, normal_const : i64) -> i64{
    value - (value * (prod + normal_const))/prod
}

#[inline(always)]
const fn intersection_xy(normal : &Normal, droite : (f32,f32,f32), point : (f32,f32,f32)) -> (f32,f32){
    let t = intersection_t(normal, droite, point);
    let x = point.0 + droite.0*t;
    let y = point.1 + droite.1*t;
    (x, y)
}

#[inline(always)]
const fn intersection_xyz(normal : &Normal, droite : (f32,f32,f32), point : (f32,f32,f32)) -> (f32,f32,f32){
    let t = intersection_t(normal, droite, point);
    let x = point.0 + droite.0*t;
    let y = point.1 + droite.1*t;
    let z = point.2 + droite.2*t;
    (x, y, z)
}

#[inline(always)]
const fn encode_i64(z : i64, color : u32) -> i64{
    (color as i64) | z << 32
}

#[allow(dead_code)]
#[inline(always)]
const fn decode(code : i64) -> (i64,u32) {
    let z = code >> 32;
    let color : u32 = code as u32;
    (z ,color)
}

#[inline(always)]
const fn same_signe(x : i64, y : i64) -> bool{
    ((x as usize ^ (y as usize)) & 0x8000000000000000) == 0
}

#[inline(always)]
const fn sub(a: (f32,f32,f32), b: (f32,f32,f32)) -> (f32,f32,f32) {
    (a.0-b.0, a.1-b.1, a.2-b.2)
}

#[inline(always)]
const fn dot(a: (f32,f32,f32), b: (f32,f32,f32)) -> f32 {
    a.0*b.0 + a.1*b.1 + a.2*b.2
}

#[inline(always)]
const fn cross(a: (f32,f32,f32), b: (f32,f32,f32)) -> (f32,f32,f32) {
    (a.1*b.2 - a.2*b.1, a.2*b.0 - a.0*b.2, a.0*b.1 - a.1*b.0)
}

const MAX_TRIANGLES : usize = 300;
const GRID_SIZE: u32 = 32;

fn split_screen<'a>(all_triangles: Vec<&'a Triangle2D>, coo : (i64, i64), grid_size : i64, gpu_buffer :  usize , gpu  : &Gpu) -> Vec<((i64,i64),i64,Vec<&'a Triangle2D>)>{
    let mut all_ptr: Vec<((i64,i64),i64,Vec<&Triangle2D>)>  = vec![];
    if all_triangles.len() > MAX_TRIANGLES && grid_size < ((WIDTH as u32 /128) as i64){
        let mut sup_screen : Vec<Vec<&Triangle2D>> = vec![vec![]; 4];
        let grid_size = grid_size << 1;
        let fac = WIDTH as i64 / grid_size;
        let cox = coo.0 * fac << 1;
        let coy = coo.1 * fac << 1;
        for tri in all_triangles.iter() {
            let case_sx = (tri.square_x + WIDTH2I) - cox;
            let case_sy = (tri.square_y + HEIGHT2I) - coy;
            let case_end_x = (tri.square_width + WIDTH2I) - cox;
            let case_end_y = (tri.square_height+ HEIGHT2I) - coy;

            if case_sx < fac && case_sy < fac && case_end_x >= 0 && case_end_y >= 0 {
                sup_screen[0].push(tri);
            }
            if case_sx < 2*fac && case_sy < fac && case_end_x >= fac && case_end_y >= 0 {
                sup_screen[1].push(tri);
            }
            if case_sx < fac && case_sy < 2*fac && case_end_x >= 0 && case_end_y >= fac {
                sup_screen[2].push(tri);
            }
            if case_sx < 2*fac && case_sy < 2*fac && case_end_x >= fac && case_end_y >= fac {
                sup_screen[3].push(tri);
            }
        }

        all_ptr.append(&mut sup_screen
            .into_par_iter()
            .enumerate()
            .flat_map(|(idx,array)| {
                let coo = (coo.0 * 2 + (idx as i64) % 2, coo.1 * 2 + (idx as i64) / 2);
                split_screen(array, coo, grid_size, gpu_buffer, gpu)
            }).collect::<Vec<((i64,i64),i64,Vec<&Triangle2D>)>>());
        all_ptr
    }else {
        vec![(coo, grid_size,all_triangles)]
    }
}

fn compute_triangle(triangles: &[Triangle3D], camera: &Camera, gpu_buffer : CUdeviceptr, gpu : &mut Gpu ) {
    let lights = camera.get_lights();
    let st = Instant::now();
    let result = triangles.par_iter().flat_map(|tri_3d| { tri_3d.to_2d(&camera)}).collect::<Vec<Triangle2D>>();
    let diff_cpu = time::Instant::now() - st.elapsed();
    let reference = result.iter().map(|tri| tri).collect::<Vec<&Triangle2D>>();
    let st = Instant::now();
    //let all_grid = split_screen(reference,(0,0),1,gpu_buffer as usize,gpu);
    let mut all_ptr : Vec<CUdeviceptr> = vec![];
    let threads : dim3 = dim3::new_2d(GRID_SIZE,GRID_SIZE);
    for (coo,grid_size,tri) in split_screen(reference,(0,0),1,gpu_buffer as usize,gpu) {
        let fac = WIDTH as i64 / grid_size;
        let block_x = ((fac as u32)+ 16 - 1) / 16;
        let mut block_y = block_x;
        if fac*(coo.1+1) > HEIGHT as i64{
            let diff = fac*(coo.1+1) - (HEIGHT as i64);
            block_y = ((diff as u32)+ 16 - 1) / 16;
        }
        let blocks  : dim3 = dim3::new_2d(block_x,block_y);
        let mut gpu_ptr: CUdeviceptr = ptr::null_mut();
        let size = tri.len();
        if tri.len() > 0{
            let triangles = tri.iter().map(|tri| {(**tri).clone()}).collect::<Vec<Triangle2D>>();
            gpu_ptr = Gpu::malloc::<Triangle2D>(size);
            Gpu::mem_copy::<Triangle2D>(gpu_ptr, triangles.as_ptr() as CUdeviceptr, size);
            all_ptr.push(gpu_ptr);
        }
        let mut params = to_params!(
                gpu_ptr as CUdeviceptr,
                gpu_buffer,
                coo.0,
                coo.1,
                grid_size,
                (WIDTH as i64),
                (HEIGHT as i64),
                size
            );
        gpu.launch_kernel(&blocks, &threads,params.as_mut_ptr());
    }
    gpu.synchronize();
    for ptr in all_ptr {
        Gpu::free(ptr);
    }
    let diff_gpu = time::Instant::now() - st.elapsed();
    println!("CPU {} GPU {}",diff_cpu.elapsed().as_nanos(), diff_gpu.elapsed().as_nanos());
}

fn main() {
    let mut gpu = Gpu::new();

    let (mut camera,triangles) = generate_config();

    //gpu.load_ptx_file(&"./gradiant.ptx".to_string());
    //gpu.get_function(&"gradiant".to_string());

    gpu.load_ptx_file(&"./color.ptx".to_string());
    gpu.get_function(&"color".to_string());

    let mut buffer: Vec<u32> = vec!(0;WIDTH*HEIGHT);
    //let triangles_gpu = Gpu::malloc::<Triangle2D>(triangles.len());
    let buffer_gpu = Gpu::malloc::<u32>(WIDTH*HEIGHT);

    //let width_b  = (WIDTH + 255 - 1) / 255;
    //let height_b = (HEIGHT + 255 - 1) / 255;
    //let threads : dim3 = dim3::new_2d(255,255);
    //let blocks  : dim3 = dim3::new_2d(width_b as u32,height_b as u32);


    println!("Triangle Len {}",triangles.len());
    // Windo management
    let mut window = Window::new("My 3D engine", WIDTH, HEIGHT, WindowOptions::default())
       .unwrap();
    window.set_target_fps(FPS);



    let mut fps : u128 = 0;
    let mut index = 0;
    let mut fps_100 = vec![0; FPS_BUFFER_SIZE];

    let mut speed : isize;
    let mut old_mouse_pos = (0f32,0f32);



    while window.is_open() && !window.is_key_down(Key::Escape) {
        camera.move_cam(&window);
        let start = Instant::now();
        compute_triangle(&triangles,&camera,buffer_gpu, &mut gpu);

        Gpu::mem_copy::<u32>(buffer.as_mut_ptr() as CUdeviceptr, buffer_gpu, WIDTH*HEIGHT);

        window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();

        let d = start.elapsed().as_nanos();
        let d2 = 1_000_000_000/d;
        unsafe {
            fps += d2.unchecked_sub(fps_100[index]);
        }
        fps_100[index] = d2;
        index = (index + 1) %FPS_BUFFER_SIZE;
        println!("\r FPS : mean {} instant {} index {} size {}",fps/FPS_BUFFER_SIZE_128,d2,index,triangles.len());
    }
    Gpu::free(buffer_gpu);
    //Gpu::free_host(buffer);
    gpu.end();
}