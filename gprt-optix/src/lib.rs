use gprt_core::{Ray, Scene, Geometry};
use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::Mutex;
use std::collections::HashMap;

#[repr(C)] pub struct CGprtPipeline { _private: [u8; 0] }
#[repr(C)] pub struct CGprtBvh { _private: [u8; 0] }

extern "C" {
    fn gprt_pipeline_create(ir_data: *const std::ffi::c_void, ir_size: usize, is_hw_triangle: i32) -> *mut CGprtPipeline;
    fn gprt_pipeline_destroy(pipe: *mut CGprtPipeline);
    fn gprt_bvh_build(pipe: *mut CGprtPipeline, h_geom: *const f32, h_aabbs: *const f32, count: i32, geom_bytes: i32) -> *mut CGprtBvh;
    fn gprt_bvh_build_triangles(pipe: *mut CGprtPipeline, h_verts: *const f32, h_indices: *const u32, num_triangles: i32, h_aabbs: *const f32) -> *mut CGprtBvh;
    fn gprt_bvh_refit(pipe: *mut CGprtPipeline, bvh: *mut CGprtBvh, h_aabbs: *const f32, count: i32);
    fn gprt_bvh_destroy(bvh: *mut CGprtBvh);
    fn gprt_register_array(pipe: *mut CGprtPipeline, name: *const c_char, capacity_per_query: u32, num_queries: i32);
    fn gprt_register_value(pipe: *mut CGprtPipeline, name: *const c_char);
    fn gprt_execute(pipe: *mut CGprtPipeline, bvh: *mut CGprtBvh, h_queries: *const f32, count: i32);
    fn gprt_retrieve_array_lengths(pipe: *mut CGprtPipeline, name: *const c_char, h_lengths: *mut u32, num_queries: i32);
    fn gprt_retrieve_array_flat(pipe: *mut CGprtPipeline, name: *const c_char, h_out: *mut u32, total_elements: i32);
    fn gprt_retrieve_value(pipe: *mut CGprtPipeline, name: *const c_char, h_out: *mut u32);
}

pub struct OptixPipeline {
    ptr: *mut CGprtPipeline,
    bvh_cache: Mutex<HashMap<u64, *mut CGprtBvh>>,
    is_hw_triangle: bool,
}
unsafe impl Send for OptixPipeline {} 
unsafe impl Sync for OptixPipeline {}

impl Drop for OptixPipeline {
    fn drop(&mut self) { 
        let cache = self.bvh_cache.lock().unwrap();
        for &bvh in cache.values() { unsafe { gprt_bvh_destroy(bvh); } }
        unsafe { gprt_pipeline_destroy(self.ptr); } 
    }
}

impl OptixPipeline {
    pub fn new(ir_bytes: &[u8], is_hw_triangle: bool) -> Self {
        Self {
            ptr: unsafe { gprt_pipeline_create(ir_bytes.as_ptr() as *const _, ir_bytes.len(), is_hw_triangle as i32) },
            bvh_cache: Mutex::new(HashMap::new()),
            is_hw_triangle,
        }
    }
    
    // THE MAGIC: Automatic Build vs Refit
    pub fn trace_scene<G: Geometry>(&self, scene: &mut Scene<G>, rays: &[Ray]) {
        let mut cache = self.bvh_cache.lock().unwrap();
        let bvh_ptr = if let Some(&bvh) = cache.get(&scene.__gprt_id) {
            if scene.__gprt_is_dirty {
                self.refit_bvh_internal(bvh, scene);
            }
            bvh
        } else {
            let bvh = self.build_bvh_internal(scene);
            cache.insert(scene.__gprt_id, bvh);
            bvh
        };
        scene.__gprt_is_dirty = false;
        self.execute_bvh(bvh_ptr, rays);
    }

    fn build_bvh_internal<G: Geometry>(&self, scene: &Scene<G>) -> *mut CGprtBvh {
        let mut aabbs: Vec<f32> = Vec::new();
        for prim in &scene.primitives {
            let b = prim.bounds();
            aabbs.extend_from_slice(&[b.min.x, b.min.y, b.min.z, b.max.x, b.max.y, b.max.z]);
        }
        let prim_count = scene.primitives.len() as i32;

        if self.is_hw_triangle {
            if let Some((verts, indices)) = G::get_hardware_triangle_buffers(&scene.primitives) {
                unsafe { gprt_bvh_build_triangles(self.ptr, verts.as_ptr(), indices.as_ptr(), prim_count, aabbs.as_ptr()) }
            } else { panic!("Hardware triangle flag set but geometry didn't provide buffers"); }
        } else {
            let mut geom: Vec<f32> = Vec::new();
            for prim in &scene.primitives { geom.extend_from_slice(&prim.pack_optix()); }
            let geom_bytes = (geom.len() * std::mem::size_of::<f32>()) as i32;
            unsafe { gprt_bvh_build(self.ptr, geom.as_ptr(), aabbs.as_ptr(), prim_count, geom_bytes) }
        }
    }

    fn refit_bvh_internal<G: Geometry>(&self, bvh: *mut CGprtBvh, scene: &Scene<G>) {
        let mut aabbs: Vec<f32> = Vec::new();
        for prim in &scene.primitives {
            let b = prim.bounds();
            aabbs.extend_from_slice(&[b.min.x, b.min.y, b.min.z, b.max.x, b.max.y, b.max.z]);
        }
        unsafe { gprt_bvh_refit(self.ptr, bvh, aabbs.as_ptr(), scene.primitives.len() as i32); }
    }

    pub fn execute_bvh(&self, bvh: *mut CGprtBvh, rays: &[Ray]) {
        let mut queries: Vec<f32> = Vec::new();
        for ray in rays { queries.extend_from_slice(&[ray.origin.x, ray.origin.y, ray.origin.z, ray.tmax]); }
        unsafe { gprt_execute(self.ptr, bvh, queries.as_ptr(), (queries.len() / 4) as i32); }
    }



pub fn retrieve_array_batched(&self, name: &str, num_queries: usize, cap_per_query: usize) -> Vec<Vec<u32>> {
    let c_name = CString::new(name).unwrap();
    let mut lengths: Vec<u32> = vec![0; num_queries];
    unsafe { gprt_retrieve_array_lengths(self.ptr, c_name.as_ptr(), lengths.as_mut_ptr(), num_queries as i32); }
    
    let total_cap = num_queries * cap_per_query;
    let mut flat_data: Vec<u32> = vec![0; total_cap];
    unsafe { gprt_retrieve_array_flat(self.ptr, c_name.as_ptr(), flat_data.as_mut_ptr(), total_cap as i32); }
    
    let mut result = Vec::with_capacity(num_queries);
    for q in 0..num_queries {
        let start = q * cap_per_query;
        // FIX: Clamp length to capacity to prevent slice overflow
        let len = (lengths[q] as usize).min(cap_per_query);
        result.push(flat_data[start..start+len].to_vec());
    }
    result
}


    
    pub fn retrieve_array(&self, name: &str, vec: &mut Vec<u32>) {
        let c_name = CString::new(name).unwrap();
        let mut single_len: Vec<u32> = vec![0; 1];
        unsafe { 
            gprt_retrieve_array_lengths(self.ptr, c_name.as_ptr(), single_len.as_mut_ptr(), 1); 
        }
        let len = single_len[0] as usize;
        if vec.capacity() < len { vec.reserve(len); }
        unsafe {
            gprt_retrieve_array_flat(self.ptr, c_name.as_ptr(), vec.as_mut_ptr(), len as i32);
            vec.set_len(len);
        }
    }
    
    pub fn retrieve_value(&self, name: &str) -> u32 {
        let c_name = CString::new(name).unwrap();
        let mut val: u32 = 0;
        unsafe { gprt_retrieve_value(self.ptr, c_name.as_ptr(), &mut val); }
        val
    }

    pub fn register_array_batched(&self, name: &str, cap_per_query: u32, num_queries: usize) {
        let c_name = CString::new(name).unwrap();
        unsafe { gprt_register_array(self.ptr, c_name.as_ptr(), cap_per_query, num_queries as i32); }
    }
    
    pub fn register_array(&self, name: &str, vec: &mut Vec<u32>) {
        let cap = vec.capacity().max(100_000) as u32;
        vec.reserve(cap as usize);
        let c_name = CString::new(name).unwrap();
        unsafe { gprt_register_array(self.ptr, c_name.as_ptr(), cap, 1); }
    }
    
    pub fn register_value(&self, name: &str, _val: &mut u32) {
        let c_name = CString::new(name).unwrap();
        unsafe { gprt_register_value(self.ptr, c_name.as_ptr()); }
    }
}
