use std::ffi::{c_void, CString};
use std::os::raw::{c_char, c_float, c_int};
use std::ptr;


#[macro_export]
macro_rules! to_params {
    ($( $val:expr ),*) => {
        [$(& ($val) as *const _ as *mut c_void,)*]
    };
}
pub(crate) use to_params;


const ERROR_PANIC : bool = true;

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUdeviceptr = *mut c_void;
pub type CUstream = *mut c_void;

#[link(name = "cuda")]
unsafe extern "C" {
    fn cuInit(flags: u32) -> CUresult;
    fn cuDeviceGet(device : *mut CUdevice , ordinal : c_int) -> CUresult;
    fn cuDeviceGetName(name : *mut u8, len : c_int, dev : CUdevice) -> CUresult;
    fn cuDeviceTotalMem_v2(bytes : *mut usize, dev : CUdevice) -> CUresult;
    fn cuDevicePrimaryCtxRetain(pctx : *mut CUcontext, dev : CUdevice) -> CUresult;
    fn cuDevicePrimaryCtxRelease(dev : CUdevice) -> CUresult;
    fn cuDevicePrimaryCtxReset(dev : CUdevice) -> CUresult;

    fn cuCtxCreate_v2(pctx : &mut CUcontext, flags: u32, dev : CUdevice) -> CUresult;
    fn cuCtxDestroy_v2(pctx : CUcontext) -> CUresult;
    fn cuCtxSetCurrent(ctx : CUcontext) -> CUresult;

    fn cuModuleLoad(module : *mut CUmodule, fname : *const c_char) -> CUresult;
    fn cuModuleLoadData(module : *mut CUmodule, image : *const c_void) -> CUresult;
    fn cuModuleGetFunction(hfunc : &mut CUfunction, hmod : CUmodule, name : *const c_char) -> CUresult;

    fn cuGetErrorString(error : CUresult, p_str : *mut *const c_char ) -> *const c_char;

    fn cuStreamCreate(ph_stream: *mut CUstream, flags : u32) -> CUresult;
    fn cuStreamDestroy_v2(h_stream : CUstream) -> CUresult;

    fn cuMemAllocHost_v2(host_p : *mut *mut c_void, bytesize : usize) -> CUresult;
    fn cuMemAlloc_v2(device_ptr : *mut CUdeviceptr, bytesize : usize) -> CUresult;
    fn cuMemAllocAsync(device_ptr : *mut CUdeviceptr, bytesize : usize, h_stream : CUstream) -> CUresult;

    fn cuMemFreeHost(p : *mut c_void) -> CUresult;
    fn cuMemFree_v2(p : CUdeviceptr) -> CUresult;
    fn cuMemFreeAsync(p : CUdeviceptr, h_stream : CUstream) -> CUresult;

    fn cuMemcpy(dst_p : CUdeviceptr, src : CUdeviceptr, bytesize : usize) -> CUresult;
    fn cuMemcpyHtoDAsync_v2(dst_p : CUdeviceptr, src: CUdeviceptr, bytesize : usize, h_stream : CUstream) -> CUresult;

    fn cuLaunchKernel(f : CUfunction, grid_dim_x : u32, grid_dim_y : u32, grid_dim_z : u32, block_dim_x : u32, block_dim_y : u32, block_dim_z : u32, shared_mem_bytes : u32, h_stream : CUstream, kernel_params : *mut *mut c_void, extra : *mut *mut c_void) -> CUresult;
    fn cuCtxSynchronize() -> CUresult;
    fn cuStreamSynchronize(stream : CUstream) -> CUresult;
}

fn check(res: CUresult, what: &str) {
    if res != 0 {
        let mut p_string : *const c_char = ptr::null_mut();
        let error : String;
        unsafe {
            cuGetErrorString(res, &mut p_string);
            error = CString::from_raw(p_string as *mut c_char).into_string().unwrap();
        }
        let message = format!("CUDA call '{what}' failed with error code {error}");
        if ERROR_PANIC{
            panic!("{}", message);
        }else{
            println!("{}", message);
            std::process::exit(1);
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug)]
pub struct dim3{
    x : u32,
    y : u32,
    z : u32,
}

impl dim3 {
    pub fn new_1d(x : u32) -> dim3{
        dim3{x, y : 1, z : 1}
    }
    pub fn new_2d(x : u32, y : u32 ) -> dim3{
        dim3{x, y, z : 1}
    }
    pub fn new_3d(x : u32, y : u32, z : u32 ) -> dim3{
        dim3{x, y, z}
    }
}

pub struct Gpu{
    device : CUdevice,
    pctx : CUcontext,
    module : CUmodule,
    function : CUfunction,
}

unsafe impl Send for Gpu{}
unsafe impl Sync for Gpu{}


impl Gpu{
    pub fn new() -> Gpu{
            let mut device: CUdevice = 0;
            //let mut device_name: [u8; 256] = [0; 256];
            //let mut device_mem_bytes: usize = 0;
            let mut pctx: CUcontext = ptr::null_mut();

            //let fname = CString::new("./vec_add.ptx").unwrap();
            //let function_name = CString::new("vecAdd2").unwrap();
            //let mut function: CUfunction = ptr::null_mut();

            //flag 0x04 = CU_CTX_SCHED_BLOCKING_SYNC (Set blocking synchronization as default scheduling)
            let ctx_flags: u32 = 0x04;
            //let ctx_flags: u32 = 0x0;
        unsafe {
            //Init Cuda
            check(cuInit(0), "cudaInit");

            //Get the GPU
            check(cuDeviceGet(&mut device, 0), "cuDeviceGet");

            //check(cuDeviceGetName(device_name.as_mut_ptr(), 256, device), "cuDeviceGetName");
            //check(cuDeviceTotalMem_v2(&mut device_mem_bytes, device), "cuDeviceGetTotalMem");
            //println!("Device {} with {} MiB", String::from_utf8(device_name.to_vec()).unwrap(), device_mem_bytes / 1024 / 1024);

            //Create the context on the gpu for the cuda program to run
            check(cuCtxCreate_v2(&mut pctx,ctx_flags,device), "cuCtxCreate");
        }
        Gpu{device, pctx, module: ptr::null_mut(), function: ptr::null_mut() }
    }
    pub fn load_ptx_file(&mut self,path : &String){
        unsafe {
            let c_str = CString::new(path.clone()).unwrap();
            //Load the ptx file
            check(cuModuleLoad(&mut self.module, c_str.as_ptr()), "cuModuleLoad");
        }
    }

    pub fn get_function(&mut self, function_name : &String){
        let c_str = CString::new(function_name.clone()).unwrap();
        unsafe {
            //Get the function needed
            check(cuModuleGetFunction(&mut self.function, self.module, c_str.as_ptr()), "cuModuleGetFunction");
        }
    }

    pub fn malloc_host<T>(size : usize) -> *mut T{
        let mut ptr = ptr::null_mut();
        unsafe {
            check(cuMemAllocHost_v2((&mut ptr) as *mut *mut T as *mut *mut c_void, size * size_of::<T>()), "cuMemAllocHost_v2");
        }
        ptr
    }

    pub fn malloc<T>(size : usize) -> CUdeviceptr{
        let mut ptr : CUdeviceptr = ptr::null_mut();
        unsafe {
            check(cuMemAlloc_v2((&mut ptr) as *mut *mut c_void , size*size_of::<T>()),"cuMemAlloc_v2");
        }
        ptr
    }

    pub fn malloc_async<T>(size : usize, stream : CUstream) -> CUdeviceptr{
        let mut ptr : CUdeviceptr = ptr::null_mut();
        unsafe {
            check(cuMemAllocAsync((&mut ptr) as *mut *mut c_void, size*size_of::<T>(), stream),"cuMemAllocAsync");
        }
        ptr
    }

    pub fn mem_copy<T>(ptr_dest : CUdeviceptr, ptr_src : CUdeviceptr, size : usize){
        unsafe {
            check(cuMemcpy(ptr_dest,ptr_src ,size*size_of::<T>()),"cuMemcpy");
        }
    }

    pub fn mem_cpy_async<T>(ptr_dest : CUdeviceptr, ptr_src: CUdeviceptr, size : usize, stream : CUstream){
        unsafe {
            check(cuMemcpyHtoDAsync_v2(ptr_dest,ptr_src,size*size_of::<T>(),stream),"cuMemcpyHtoDAsync");
        }
    }

    pub fn free(ptr: CUdeviceptr){
        unsafe {
            check(cuMemFree_v2(ptr),"cuMemFree");
        }
    }

    pub fn free_async(ptr: CUdeviceptr, stream : CUstream){
        unsafe {
            check(cuMemFreeAsync(ptr,stream),"cuMemFreeAsync");
        }
    }

    pub fn free_host<T>(ptr: *mut T){
        unsafe {
            check(cuMemFreeHost(ptr as *mut c_void),"cuMemFreeHost");
        }
    }

    pub fn destroy_ctx(&self){
        unsafe {
            check(cuCtxDestroy_v2(self.pctx), "cuCtxDestroy");
        }
    }

    pub fn set_ctx(&self){
        unsafe {
            check(cuCtxSetCurrent(self.pctx), "cuCtxSetCurrent");
        }
    }

    pub fn synchronize(&self){
        unsafe {
            check(cuCtxSynchronize(), "cuCtxSynchronize");
        }
    }

    pub fn synchronize_stream(stream : CUstream){
        unsafe {
            check(cuStreamSynchronize(stream), "cuStreamSynchronize");
        }
    }

    pub fn launch_kernel(&self, blocks : &dim3, thread : &dim3 , kernel_params : *mut *mut c_void,stream : CUstream){
        unsafe {
            check(cuLaunchKernel(self.function,blocks.x,blocks.y,blocks.z, thread.x,thread.y,thread.z, 0, stream,kernel_params,ptr::null_mut()),"cuLaunchKernel");
        }
    }

    pub fn create_steam() -> CUstream{
        let mut st : CUstream = ptr::null_mut();
        unsafe {
            //flags = CU_STREAM_NON_BLOCKING = 0x01
            check(cuStreamCreate(&mut st,0x01),"cuStreamCreate");
        }
        st
    }

    pub fn destroy_stream(stream : CUstream){
        unsafe {
            check(cuStreamDestroy_v2(stream), "cuStreamDestroy");
        }
    }

    pub fn end(&self){
        self.destroy_ctx();
    }
}

