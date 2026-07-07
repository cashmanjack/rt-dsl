use gprt_core::{Ray, Scene, Geometry};
use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::Mutex;
use std::collections::HashMap;
use gprt_ir::Schedule;

pub mod dispatch;

#[repr(C)] pub struct CGprtPipeline { _private: [u8; 0] }
#[repr(C)] pub struct CGprtBvh { _private: [u8; 0] }

extern "C" {
    fn gprt_pipeline_create(ir_data: *const std::ffi::c_void, ir_size: usize, is_hw_triangle: i32) -> *mut CGprtPipeline;
    fn gprt_pipeline_destroy(pipe: *mut CGprtPipeline);
    fn gprt_bvh_build(pipe: *mut CGprtPipeline, h_geom: *const f32, h_aabbs: *const f32, count: i32, geom_bytes: i32) -> *mut CGprtBvh;
    fn gprt_bvh_build_triangles(pipe: *mut CGprtPipeline, h_verts: *const f32, h_indices: *const u32, num_triangles: i32, h_aabbs: *const f32) -> *mut CGprtBvh;
    fn gprt_bvh_refit(pipe: *mut CGprtPipeline, bvh: *mut CGprtBvh, radius: f32, count: i32);    
    fn gprt_bvh_destroy(bvh: *mut CGprtBvh);
    fn gprt_register_array(pipe: *mut CGprtPipeline, name: *const c_char, capacity_per_query: u32, num_queries: i32);
    fn gprt_register_value(pipe: *mut CGprtPipeline, name: *const c_char);
    fn gprt_execute(pipe: *mut CGprtPipeline, bvh: *mut CGprtBvh, h_queries: *const f32, count: i32);
    fn gprt_retrieve_array_lengths(pipe: *mut CGprtPipeline, name: *const c_char, h_lengths: *mut u32, num_queries: i32);
    fn gprt_retrieve_array_flat(pipe: *mut CGprtPipeline, name: *const c_char, h_out: *mut u32, total_elements: i32);
    fn gprt_retrieve_value(pipe: *mut CGprtPipeline, name: *const c_char, h_out: *mut u32);
    fn gprt_execute_autorope_soa_ffi(
        pipe: *mut CGprtPipeline, h_bodies: *const std::ffi::c_void, num_bodies: i32,
        h_spatial: *const std::ffi::c_void, num_nodes: i32, h_routing: *const std::ffi::c_void,
        theta: f32, h_out_forces: *mut std::ffi::c_void
    );
}

pub struct OptixPipeline {
    ptr: *mut CGprtPipeline,
    bvh_cache: Mutex<HashMap<u64, *mut CGprtBvh>>,
    is_hw_triangle: bool,
    registered_arrays: Mutex<HashMap<String, (u32, usize)>>,
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
            registered_arrays: Mutex::new(HashMap::new()),
        }
    }




    pub fn trace_scene<G: Geometry>(&self, scene: &mut Scene<G>, rays: &[Ray], _schedule: &Schedule) {
        let mut cache = self.bvh_cache.lock().unwrap();
        let bvh_ptr = if let Some(&bvh) = cache.get(&scene.__gprt_id) {
            if scene.__gprt_is_dirty { 
                drop(cache);
                self.refit_bvh_internal(bvh, scene); // FIXED: Added missing bvh argument
                cache = self.bvh_cache.lock().unwrap();
            }
            *cache.get(&scene.__gprt_id).unwrap()
        } else {
            drop(cache);
            self.build_bvh_internal(scene)
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
                let bvh_ptr = unsafe { gprt_bvh_build_triangles(self.ptr, verts.as_ptr(), indices.as_ptr(), prim_count, aabbs.as_ptr()) };
                self.bvh_cache.lock().unwrap().insert(scene.__gprt_id, bvh_ptr);
                bvh_ptr
            } else { panic!("Hardware triangle flag set but geometry didn't provide buffers"); }
        } else {
            let mut geom: Vec<f32> = Vec::new();
            for prim in &scene.primitives { geom.extend_from_slice(&prim.pack_optix()); }
            let geom_bytes = (geom.len() * std::mem::size_of::<f32>()) as i32;
            let bvh_ptr = unsafe { gprt_bvh_build(self.ptr, geom.as_ptr(), aabbs.as_ptr(), prim_count, geom_bytes) };
            self.bvh_cache.lock().unwrap().insert(scene.__gprt_id, bvh_ptr);
            bvh_ptr
        }
    }



    fn refit_bvh_internal<G: Geometry>(&self, bvh: *mut CGprtBvh, scene: &Scene<G>) {
        // FIXED: Extract radius from the first primitive's bounds instead of assuming a .radius field
        let b0 = scene.primitives[0].bounds();
        let radius = (b0.max.x - b0.min.x) / 2.0;
        unsafe { gprt_bvh_refit(self.ptr, bvh, radius, scene.primitives.len() as i32); }
    }


    pub fn execute_bvh(&self, bvh: *mut CGprtBvh, rays: &[Ray]) {
        let mut queries: Vec<f32> = Vec::new();
        for ray in rays { queries.extend_from_slice(&[ray.origin.x, ray.origin.y, ray.origin.z, ray.tmax]); }
        unsafe { gprt_execute(self.ptr, bvh, queries.as_ptr(), (queries.len() / 4) as i32); }
    }

    pub fn retrieve_array_batched(&self, name: &str, num_queries: usize, cap_per_query: usize) -> (Vec<u32>, Vec<u32>) {
        let c_name = CString::new(name).unwrap();
        let mut lengths: Vec<u32> = vec![0; num_queries];
        unsafe { gprt_retrieve_array_lengths(self.ptr, c_name.as_ptr(), lengths.as_mut_ptr(), num_queries as i32); }
        let total_cap = num_queries * cap_per_query;
        let mut flat_data: Vec<u32> = vec![0; total_cap];
        unsafe { gprt_retrieve_array_flat(self.ptr, c_name.as_ptr(), flat_data.as_mut_ptr(), total_cap as i32); }
        (flat_data, lengths)
    }

    pub fn register_array_batched(&self, name: &str, cap_per_query: u32, num_queries: usize) {
        let mut cache = self.registered_arrays.lock().unwrap();
        let c_name = CString::new(name).unwrap();
        unsafe { gprt_register_array(self.ptr, c_name.as_ptr(), cap_per_query, num_queries as i32); }
        cache.insert(name.to_string(), (cap_per_query, num_queries));
    }
    
    pub fn clear_bvh_cache(&self) {
        let mut cache = self.bvh_cache.lock().unwrap();
        for &bvh in cache.values() { unsafe { gprt_bvh_destroy(bvh); } }
        cache.clear();
    }


    pub fn execute_autorope_soa(&self, bodies: &[gprt_core::Body], spatial: &[gprt_core::NodeSpatial], routing: &[gprt_core::NodeRouting], theta: f32) -> Vec<gprt_core::Vec3> {
        let num_bodies = bodies.len() as i32;
        let num_nodes = spatial.len() as i32;
        let mut out_forces: Vec<gprt_core::Vec3> = vec![gprt_core::Vec3::new(0.0, 0.0, 0.0); num_bodies as usize];
        unsafe {
            gprt_execute_autorope_soa_ffi(
                self.ptr, 
                bodies.as_ptr() as *const std::ffi::c_void, num_bodies, 
                spatial.as_ptr() as *const std::ffi::c_void, num_nodes, 
                routing.as_ptr() as *const std::ffi::c_void, 
                theta, 
                out_forces.as_mut_ptr() as *mut std::ffi::c_void
            );
        }
        out_forces
    }

}

pub struct IndexBuilder<G: Geometry> {
    pub scene: Scene<G>,
    pub schedule: Schedule,
}

impl<G: Geometry> IndexBuilder<G> {
    pub fn compile_and_bind(self, ir_bytes: &[u8]) -> SpatialIndex<G> {
        let pipeline = OptixPipeline::new(ir_bytes, false);
        SpatialIndex { pipeline, scene: self.scene, schedule: self.schedule }
    }
}

pub struct SpatialIndex<G: Geometry> {
    pub pipeline: OptixPipeline,
    pub scene: Scene<G>,
    pub schedule: Schedule,
}

impl<G: Geometry> SpatialIndex<G> {
    pub fn mark_dirty(&mut self) { self.scene.mark_dirty(); }
    pub fn execute_trace(&mut self, queries: &[Ray], schedule: &Schedule) {
        self.pipeline.trace_scene(&mut self.scene, queries, schedule);
    }
}
