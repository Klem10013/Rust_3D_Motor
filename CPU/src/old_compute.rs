fn compute_triangle_bad(triangle_2d : &Triangle2D, camera : &Camera, grid : &mut Vec<(u32,usize)>) -> (){
    let mut x : usize = 0;
    let mut y : usize = 0;
    let mut width : usize = WIDTH;
    let mut height : usize = HEIGHT;
    for i in x..(width+x){
        for j in y..(height+y){
            let d = distance_squared((i,j),triangle_2d.p1);
            if d < triangle_2d.p1p2 && d < triangle_2d.p3p1{
                let d = distance_squared((i,j),triangle_2d.p2);
                if d < triangle_2d.p1p2 && d < triangle_2d.p2p3{
                    let d = distance_squared((i,j),triangle_2d.p3);
                    if d < triangle_2d.p2p3 && d < triangle_2d.p3p1 {
                        grid[j*WIDTH+i] = ((i+j) as u32,0)
                    }
                }
            }
        }
    }
}

fn compute_triangle_all_screen(triangle_2d : &Vec<Triangle2D>,camera: &Camera, grid : &mut [u32]) -> (){
    grid.par_iter_mut().enumerate().for_each(|(i,pixel)| {
        let x = (i % WIDTH) as isize;
        let y = (i / WIDTH) as isize;
        for tri in triangle_2d.iter(){
            if x < tri.square_x || x > tri.square_width || y < tri.square_y || y > tri.square_height{
                continue;
            }
            let d1: isize = sign((x, y), tri.p1i, tri.p2i);
            let d2: isize = sign((x, y), tri.p2i, tri.p3i);
            let d3: isize = sign((x, y), tri.p3i, tri.p1i);
            let has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
            let has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
            if !(has_neg && has_pos) {
                *pixel = 0x00FF00;
            }
        }
    })
}