use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Expr, ExprClosure, Token, parse::{Parse, ParseStream}, LitStr};
use std::collections::HashSet;
use syn::visit::Visit;

// ==========================================
// PARSERS
// ==========================================
struct BuildIndexInput { data: Expr, geom: Expr, schedule: Option<Expr> }
impl Parse for BuildIndexInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let data: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let geom: Expr = input.parse()?;
        let schedule = if input.peek(Token![,]) { input.parse::<Token![,]>()?; Some(input.parse::<Expr>()?) } else { None };
        Ok(BuildIndexInput { data, geom, schedule })
    }
}

struct TraceInput { index: Expr, rays: Expr, on_hit: ExprClosure }
impl Parse for TraceInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let index: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let rays: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let on_hit: ExprClosure = input.parse()?;
        Ok(TraceInput { index, rays, on_hit })
    }
}

struct TraceBatchedInput { index: Expr, rays: Expr, cap: Expr }
impl Parse for TraceBatchedInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let index: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let rays: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let cap: Expr = input.parse()?;
        Ok(TraceBatchedInput { index, rays, cap })
    }
}

struct RnnInput { data: Expr, queries: Expr, radius: Expr, output: Expr }
impl Parse for RnnInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let data: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let queries: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let radius: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let output: Expr = input.parse()?;
        Ok(RnnInput { data, queries, radius, output })
    }
}


struct RnnRunInput { index: Expr, rays: Expr, schedule: Expr, cap: Expr, name: syn::Ident }
impl Parse for RnnRunInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let index: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let rays: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let schedule: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let cap: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let name: syn::Ident = input.parse()?; 
        Ok(RnnRunInput { index, rays, schedule, cap, name })
    }
}


struct KnnInput { data: Expr, queries: Expr, k: Expr, output: Expr, schedule: Option<Expr> }
impl Parse for KnnInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let data: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let queries: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let k: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let output: Expr = input.parse()?;
        let schedule = if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            Some(input.parse::<Expr>()?)
        } else { None };
        Ok(KnnInput { data, queries, k, output, schedule })
    }
}



struct AutotuneInput { data: Expr, queries: Expr, k: Expr }
impl Parse for AutotuneInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let data: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let queries: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let k: Expr = input.parse()?;
        Ok(AutotuneInput { data, queries, k })
    }
}




struct BarnesHutInput { bodies: Expr, theta: Expr, g_const: Expr, output: Expr }
impl Parse for BarnesHutInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let bodies: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let theta: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let g_const: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let output: Expr = input.parse()?;
        Ok(BarnesHutInput { bodies, theta, g_const, output })
    }
}

struct PayloadVisitor { pub arrays: HashSet<String> }
impl<'ast> Visit<'ast> for PayloadVisitor {
    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        if node.method == "push" {
            if let syn::Expr::Path(p) = &*node.receiver {
                if let Some(ident) = p.path.get_ident() { self.arrays.insert(ident.to_string()); }
            }
        }
        syn::visit::visit_expr_method_call(self, node);
    }
}



#[proc_macro]
pub fn gprt_autotune(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as AutotuneInput);
    let data = input.data;
    let queries = input.queries;
    let k = input.k;

    let expanded = quote! {
        {
            println!("\n=========================================");
            println!("   [GPRT AUTOTUNE] Adaptive Hill-Climbing Search");
            println!("=========================================");
            
            let __full_data = &#data;
            let __full_queries = &#queries;
            
            // Start at a mathematically safe baseline
            let mut __best_p: f32 = 0.01;
            let mut __best_m: f32 = 3.0;
            let mut __best_time = std::time::Duration::MAX;
            
            // Define the step sizes for the adaptive search
            let __p_steps = [-0.005, 0.005, -0.01, 0.05];
            let __m_steps = [-0.5, 0.5, -1.0, 1.0];
            
            let mut __improved = true;
            let mut __iterations = 0;
            
            while __improved && __iterations < 5 {
                __improved = false;
                __iterations += 1;
                println!("   -> Hill-Climb Iteration {}", __iterations);
                
                // Test P perturbations
                for &__dp in &__p_steps {
		    let __test_p = (__best_p + __dp).max(0.001).min(0.20); 
                    let mut __sched = gprt_ir::Schedule::default();
                    __sched.radius_heuristic = gprt_ir::RadiusHeuristic::SampledPercentile(__test_p);
                    __sched.radius_increment_mult = __best_m;
                    __sched.memory_strategy = gprt_ir::MemoryStrategy::PayloadRegisterHeap;
                    __sched.use_morton_lbv = true;
                    
                    let mut __out: Vec<Vec<u32>> = Vec::new();
                    let __t0 = std::time::Instant::now();
                    gprt_macros::k_nn!(__full_data, __full_queries, #k, __out, __sched);
                    let __elapsed = __t0.elapsed();
                    
                    let __success = __out.len() == __full_queries.len() && __out.iter().all(|res| res.len() >= #k);
                    if __success && __elapsed < __best_time {
                        println!("      [CLIMB] P={:.3}, M={:.1} | Time: {:.3}ms (New Best!)", __test_p, __best_m, __elapsed.as_secs_f64() * 1000.0);
                        __best_time = __elapsed;
                        __best_p = __test_p;
                        __improved = true;
                    }
                }
                
                // Test M perturbations
                for &__dm in &__m_steps {
                    let __test_m = (__best_m + __dm).max(1.5).min(5.0);
                    let mut __sched = gprt_ir::Schedule::default();
                    __sched.radius_heuristic = gprt_ir::RadiusHeuristic::SampledPercentile(__best_p);
                    __sched.radius_increment_mult = __test_m;
                    __sched.memory_strategy = gprt_ir::MemoryStrategy::PayloadRegisterHeap;
                    __sched.use_morton_lbv = true;
                    
                    let mut __out: Vec<Vec<u32>> = Vec::new();
                    let __t0 = std::time::Instant::now();
                    gprt_macros::k_nn!(__full_data, __full_queries, #k, __out, __sched);
                    let __elapsed = __t0.elapsed();
                    
                    let __success = __out.len() == __full_queries.len() && __out.iter().all(|res| res.len() >= #k);
                    if __success && __elapsed < __best_time {
                        println!("      [CLIMB] P={:.3}, M={:.1} | Time: {:.3}ms (New Best!)", __best_p, __test_m, __elapsed.as_secs_f64() * 1000.0);
                        __best_time = __elapsed;
                        __best_m = __test_m;
                        __improved = true;
                    }
                }
            }
            
            let mut __final_schedule = gprt_ir::Schedule::default();
            __final_schedule.radius_heuristic = gprt_ir::RadiusHeuristic::SampledPercentile(__best_p);
            __final_schedule.radius_increment_mult = __best_m;
            __final_schedule.memory_strategy = gprt_ir::MemoryStrategy::PayloadRegisterHeap;
            __final_schedule.use_morton_lbv = true;

            println!("   -> Optimum Reached: P={:.3}, M={:.1} (Time: {:.3}ms)", __best_p, __best_m, __best_time.as_secs_f64() * 1000.0);
            println!("=========================================\n");
            
            let __final_ret = __final_schedule;
            __final_ret
        }
    };
    TokenStream::from(expanded)
}



// ==========================================
// 2. STATEFUL BUILD (LICM)
// ==========================================
#[proc_macro]
pub fn gprt_build_index(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as BuildIndexInput);
    let data = input.data;
    let geom = input.geom;
    let schedule_expr = input.schedule.unwrap_or_else(|| syn::parse_str("gprt_ir::Schedule::default()").unwrap());

    let expanded = quote! {
        {
            let __geom_fn = #geom;
            let __scene = gprt_core::Scene::build(#data.iter().map(|__p| __geom_fn(__p)));
            let __schedule: gprt_ir::Schedule = #schedule_expr;
            gprt_optix::IndexBuilder { scene: __scene, schedule: __schedule }
        }
    };
    TokenStream::from(expanded)
}







#[proc_macro]
pub fn gprt_rnn_run(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as RnnRunInput);
    let index = input.index;
    let rays = input.rays;
    let schedule = input.schedule;
    let cap = input.cap;
    let name = input.name.to_string();

    let expanded = quote! {
        {
            static __GPRT_RNN_PIPELINE_CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<u32, gprt_optix::OptixPipeline>>> = std::sync::OnceLock::new();
            let __cache = __GPRT_RNN_PIPELINE_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
            
            let __cap_val: u32 = #cap;
            let __schedule_clone = #schedule.clone();
            
            let mut __map = __cache.lock().unwrap();
            
            if !__map.contains_key(&__cap_val) {
                let __is_register_heap = __schedule_clone.memory_strategy == gprt_ir::MemoryStrategy::PayloadRegisterHeap;
                let __k_str = __cap_val.to_string(); 
                
                let __struct_def = format!(r#"
                    struct LocalHeap {{
                        unsigned int ids[{}];
                        float dists[{}];
                        float worst_dist;
                    }};
                "#, __k_str, __k_str);

                let __raygen_cuda_global = r#"
                    uint3 launch_idx = optixGetLaunchIndex(); int idx = launch_idx.x; if (idx >= params.num_rays) return;
                    PayloadBundle* bundle = params.bundle;
                    ((unsigned int*)bundle->dyn_lens[0])[idx] = 0;
                    float r_val = params.rays[idx].w;
                    float3 origin = make_float3(params.rays[idx].x, params.rays[idx].y, params.rays[idx].z); 
                    float3 direction = make_float3(1.0f, 0.0f, 0.0f);
                    unsigned int p0 = 0, p1 = 0xFFFFFFFF, p2 = __float_as_uint(r_val * r_val), p3 = 0; 
                    optixTrace(params.handle, origin, direction, 0.0f, 1e-5f, 0.0f, 1u, OPTIX_RAY_FLAG_NONE, 0u, 1u, 0u, p0, p1, p2, p3);
                "#.to_string();
                
                let __anyhit_cuda_global = r#"
                    unsigned int prim_id = optixGetPrimitiveIndex();
                    PayloadBundle* bundle = params.bundle; unsigned int __qid = optixGetLaunchIndex().x;
                    unsigned int __idx = atomicAdd((unsigned int*)bundle->dyn_lens[0] + __qid, 1u);
                    if (__idx < bundle->dyn_caps[0]) {
                        ((unsigned int*)bundle->dyn_ptrs[0])[__qid * bundle->dyn_caps[0] + __idx] = prim_id;
                    }
                    optixIgnoreIntersection();
                "#.to_string();






                let k = __cap_val as usize; // This is 2*K
                let half_k = k / 2;         // This is K

                let __raygen_cuda_register_heap = format!(r#"
                    uint3 launch_idx = optixGetLaunchIndex(); 
                    int idx = launch_idx.x; 
                    if (idx >= params.num_rays) return;
                    PayloadBundle* bundle = params.bundle;

                    // Layout: [ID0, ID1, ..., ID_K-1, Dist0, Dist1, ..., Dist_K-1]
                    unsigned int* heap = &((unsigned int*)bundle->dyn_ptrs[0])[idx * {}];

                    float r_val = params.rays[idx].w;
                    float r_sq = r_val * r_val;

                    for (int i = 0; i < {}; i++) {{
                        heap[i] = 0xFFFFFFFF;
                        heap[{} + i] = __float_as_uint(r_sq);
                    }}

                    float3 origin = make_float3(params.rays[idx].x, params.rays[idx].y, params.rays[idx].z); 
                    float3 direction = make_float3(1.0f, 0.0f, 0.0f);

                    // p2 holds the dynamic worst_dist for hardware pruning!
                    unsigned int p0 = 0, p1 = 0, p2 = __float_as_uint(r_sq), p3 = 0; 
                    optixTrace(params.handle, origin, direction, 0.0f, 1e-5f, 0.0f, 1u, OPTIX_RAY_FLAG_NONE, 0u, 1u, 0u, p0, p1, p2, p3);

                    unsigned int count = 0;
                    for (int i = 0; i < {}; i++) {{ if (heap[i] != 0xFFFFFFFF) count++; }}
                    ((unsigned int*)bundle->dyn_lens[0])[idx] = count;
                "#, k, half_k, half_k, half_k);

                let __intersection_cuda = r#"
                    unsigned int prim_id = optixGetPrimitiveIndex();
                    float4 sphere = params.geom[prim_id];
                    float3 o = optixGetWorldRayOrigin();
                    float dx = o.x - sphere.x; float dy = o.y - sphere.y; float dz = o.z - sphere.z;
                    float dist_sq = dx*dx + dy*dy + dz*dz;
                    float search_radius_sq = __uint_as_float(optixGetPayload_2());
                    if (dist_sq <= search_radius_sq) {
                        optixSetPayload_3(__float_as_uint(dist_sq));
                        optixReportIntersection(1e-6f, 0u);
                    }
                "#.to_string();

                let __anyhit_cuda_register_heap = format!(r#"
                    unsigned int prim_id = optixGetPrimitiveIndex();
                    float exact_dist_sq = __uint_as_float(optixGetPayload_3());

                    unsigned int __qid = optixGetLaunchIndex().x;
                    PayloadBundle* bundle = params.bundle;
                    unsigned int* heap = &((unsigned int*)bundle->dyn_ptrs[0])[__qid * {}];

                    float worst_dist = __uint_as_float(heap[{}]); 

                    if (exact_dist_sq < worst_dist) {{
                        heap[{}] = prim_id;
                        heap[{}] = __float_as_uint(exact_dist_sq);

                        for (int i = {} - 2; i >= 0; i--) {{
                            float d_curr = __uint_as_float(heap[{} + i]);
                            float d_next = __uint_as_float(heap[{} + i + 1]);
                            if (d_curr > d_next) {{
                                heap[{} + i] = __float_as_uint(d_next);
                                heap[{} + i + 1] = __float_as_uint(d_curr);
                                unsigned int tmp_id = heap[i];
                                heap[i] = heap[i + 1];
                                heap[i + 1] = tmp_id;
                            }} else {{
                                break;
                            }}
                        }}
                        optixSetPayload_2(heap[{}]); 
                    }}
                    optixIgnoreIntersection();
                "#, k, k - 1, half_k - 1, k - 1, half_k, half_k, half_k, half_k, half_k, k - 1);

                let __rg = if __is_register_heap { __raygen_cuda_register_heap } else { __raygen_cuda_global };
                let __ah = if __is_register_heap { __anyhit_cuda_register_heap } else { __anyhit_cuda_global };

                let __ir = gprt_ir::RtProgram {
                    raygen_body: gprt_ir::ShaderNode::RawCuda(__rg),
                    anyhit_body: Some(gprt_ir::ShaderNode::RawCuda(__ah)),
                    miss_body: None, closesthit_body: None, 
                    intersection_body: Some(gprt_ir::ShaderNode::RawCuda(__intersection_cuda)),
                    payload_layout: vec![], schedule: __schedule_clone, array_indices: std::collections::HashMap::new(),
                };


                
                let __pipeline = gprt_optix::OptixPipeline::new(&gprt_codegen::compile_program(&__ir), false);
                __map.insert(__cap_val, __pipeline);
            }
            
            let __pipeline = __map.get(&__cap_val).unwrap();
            let __num_rays = #rays.len();
            let __name_str: &str = #name;
            
            __pipeline.register_array_batched(__name_str, __cap_val, __num_rays);
            __pipeline.trace_scene(&mut #index.scene, &#rays, &#schedule); 
            __pipeline.retrieve_array_batched(__name_str, __num_rays, __cap_val as usize)
        }
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn k_nn(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as KnnInput);
    let data = input.data;
    let queries = input.queries;
    let k = input.k;
    let output = input.output;
    let schedule = input.schedule.unwrap_or_else(|| syn::parse_str("gprt_ir::Schedule::default()").unwrap());

    let expanded = quote! {
        {
            #output.clear();
            #output.resize(#queries.len(), Vec::new());
            
            let __schedule = #schedule;
            let mut __current_radius = 1.0; 
            
            match __schedule.radius_heuristic {
                gprt_ir::RadiusHeuristic::Fixed(r) => __current_radius = r,
                gprt_ir::RadiusHeuristic::SampledPercentile(p) => {
                    let mut __dists = Vec::new();
                    let __q_step = (#queries.len() / 100).max(1);
                    for __qi in (0..#queries.len()).step_by(__q_step).take(100) {
                        let __q = #queries[__qi];
                        let mut __knn = vec![f32::MAX; #k];
                        let __p_step = (#data.len() / 1000).max(1);
                        for __pi in (0..#data.len()).step_by(__p_step).take(1000) {
                            let __d2 = (__q.x - #data[__pi].x).powi(2) + (__q.y - #data[__pi].y).powi(2) + (__q.z - #data[__pi].z).powi(2);
                            if __d2 > 1e-10 && __d2 < __knn[#k - 1] {
                                __knn[#k - 1] = __d2;
                                __knn.sort_by(|a, b| a.partial_cmp(b).unwrap());
                            }
                        }
                        if __knn[#k - 1] != f32::MAX { __dists.push(__knn[#k - 1].sqrt()); }
                    }
                    if !__dists.is_empty() {
                        __dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let __idx = ((__dists.len() as f32 * p) as usize).min(__dists.len() - 1);
                        __current_radius = __dists[__idx] * 1.1; 
                    }
                }
                _ => {}
            }
            println!("[GPRT_LOOP] Initial Radius Calculated: {:.5}", __current_radius);
            
            let __scene = gprt_core::Scene::build(#data.iter().map(|__p| gprt_core::Sphere { center: *__p, radius: __current_radius }));
            let mut __index = gprt_optix::IndexBuilder { scene: __scene, schedule: __schedule.clone() };            

            let mut __active_indices: Vec<usize> = if __schedule.use_morton_lbv {
                gprt_core::morton::sort_by_morton(&#queries)
            } else {
                (0..#queries.len()).collect()
            };

            let mut __iteration = 0;
            let mut __total_search_time = std::time::Duration::ZERO;
            let mut __total_saturated_registered = 0;
            
            while !__active_indices.is_empty() && __iteration < 30 {
                __iteration += 1;
                
                let mut __dynamic_cap = 128u32; 

                if __schedule.memory_strategy == gprt_ir::MemoryStrategy::PayloadRegisterHeap {
                    __dynamic_cap = (#k * 2) as u32; 
                } else {
                    if __iteration == 2 { __dynamic_cap = 256; }
                    if __iteration == 3 { __dynamic_cap = 512; }
                    if __iteration == 4 { __dynamic_cap = 1024; }
                    if __iteration == 5 { __dynamic_cap = 2048; }
                    if __iteration == 6 { __dynamic_cap = 4096; }
                    if __iteration >= 7 { __dynamic_cap = 10000; }

                    let __safe_cap = (50_000_000usize / __active_indices.len()).max(128) as u32;
                    __dynamic_cap = __dynamic_cap.min(__safe_cap);
                }

                let __rays: Vec<gprt_core::Ray> = __active_indices.iter()
                    .map(|&qi| gprt_core::Ray::query(#queries[qi], __current_radius))
                    .collect();
                




                let __t_trace = std::time::Instant::now();
                let __cap_u32: u32 = __dynamic_cap;

                let (__flat_res, __lens) = gprt_macros::gprt_rnn_run!(__index, __rays, __schedule, __cap_u32, out);

                // ==========================================
                // DEBUG PRINTS: RAW VRAM DUMP
                // ==========================================
                if __iteration == 1 {
                    println!("\n[DEBUG ROUND 1] GPU -> CPU Raw Memory Dump (Cap={}):", __cap_u32);
                    for dbg_idx in 0..3 {
                        let start = dbg_idx * (__cap_u32 as usize);
                        let end = start + (__cap_u32 as usize);
                        let lens_val = __lens.get(dbg_idx).unwrap_or(&999);
                        let slice = if end <= __flat_res.len() { &__flat_res[start..end] } else { &[] };
                        println!("  Query {}: Lens={}, RawHeap={:?}", dbg_idx, lens_val, slice);
                    }
                    println!("-------------------------------------------------\n");
                }

                __total_search_time += __t_trace.elapsed();



                
                let mut __next_active = Vec::new();
                let mut __saturated_count = 0;
                
                if __schedule.memory_strategy == gprt_ir::MemoryStrategy::PayloadRegisterHeap {


                    for (local_idx, &global_qi) in __active_indices.iter().enumerate() {
                        let hit_count = __lens[local_idx] as usize;
                        let start = local_idx * (#k as usize * 2); 

                        if hit_count >= #k {
                            for __i in 0..#k { #output[global_qi].push(__flat_res[start + __i]); }
                        } else if __iteration >= 15 {
                            for __i in 0..hit_count { #output[global_qi].push(__flat_res[start + __i]); }
                        } else {
                            __next_active.push(global_qi);
                        }
                    }


                } else {
                    use rayon::prelude::*;
                    let __results_updates: Vec<(usize, Vec<u32>, bool, bool)> = (0..__active_indices.len())
                        .into_par_iter()
                        .map(|__local_idx| {
                            let __global_qi = __active_indices[__local_idx];
                            let __raw_count = __lens[__local_idx] as usize;
                            let __is_saturated = __raw_count >= __dynamic_cap as usize && (__dynamic_cap as usize) < #data.len();
                            
                            let __hit_count = __raw_count.min(__dynamic_cap as usize);
                            let __start = __local_idx * __dynamic_cap as usize;
                            
                            let mut __hits_with_dist = Vec::with_capacity(__hit_count);
                            for __i in 0..__hit_count {
                                let __id = __flat_res[__start + __i];
                                let __p = #data[__id as usize];
                                let __q = #queries[__global_qi];
                                let __d2 = (__q.x - __p.x).powi(2) + (__q.y - __p.y).powi(2) + (__q.z - __p.z).powi(2);
                                __hits_with_dist.push((__id, __d2));
                            }
                            
                            __hits_with_dist.sort_by(|__a, __b| __a.1.partial_cmp(&__b.1).unwrap());
                            __hits_with_dist.dedup_by(|__a, __b| __a.0 == __b.0);

                            let mut __res = Vec::new();
                            let mut __is_active = true;

                            if !__is_saturated && __hits_with_dist.len() >= #k {
                                for __i in 0..#k { __res.push(__hits_with_dist[__i].0); }
                                __is_active = false;
                            } else if __is_saturated && __iteration >= 15 {
                                let __take = __hits_with_dist.len().min(#k);
                                for __i in 0..__take { __res.push(__hits_with_dist[__i].0); }
                                __is_active = false;
                            }

                            (__global_qi, __res, __is_active, __is_saturated)
                        })
                        .collect();
                    
                    for (__global_qi, __res, __is_active, __is_saturated) in __results_updates {
                        if !__res.is_empty() { #output[__global_qi] = __res; }
                        if __is_active { __next_active.push(__global_qi); }
                        if __is_saturated { __saturated_count += 1; }
                    }
                }
                
                println!("[GPRT_LOOP] Round {}: Active={}, Radius={:.5}, Saturated={}/{}, Cap={}", 
                         __iteration, __active_indices.len(), __current_radius, __saturated_count, __active_indices.len(), __dynamic_cap);
                
                __total_saturated_registered += __saturated_count;
                __active_indices = __next_active;
                if __active_indices.is_empty() { break; }
                
                __current_radius *= __schedule.radius_increment_mult;
                for prim in &mut __index.scene.primitives { prim.radius = __current_radius; }
                __index.scene.mark_dirty(); 
            }
            
            let __sat_ratio = if #queries.len() > 0 { (__total_saturated_registered as f64) / (#queries.len() as f64) } else { 0.0 };
            println!("[GPRT_STATS] search_ms={:.3}, saturation={:.2}%", 
                     __total_search_time.as_secs_f64() * 1000.0, 
                     __sat_ratio * 100.0);
        }
    };
    TokenStream::from(expanded)
}

















// ==========================================
// 5. LEGACY MACROS (RETAINED)
// ==========================================
#[proc_macro]
pub fn gprt_trace(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as TraceInput);
    let index = input.index;
    let rays = input.rays;
    let on_hit = input.on_hit;

    let mut visitor = PayloadVisitor { arrays: HashSet::new() };
    visitor.visit_expr_closure(&on_hit);
    let array_name = visitor.arrays.into_iter().next().unwrap_or_else(|| "out_array".to_string());
    let array_ident: Expr = syn::parse_str(&array_name).unwrap();

    let raygen_cuda = r#"
        uint3 launch_idx = optixGetLaunchIndex(); int idx = launch_idx.x; if (idx >= params.num_rays) return;
        float4 r = params.rays[idx]; float3 origin = make_float3(r.x, r.y, r.z); float3 direction = make_float3(1.0f, 0.0f, 0.0f);
        unsigned int p0 = 0; unsigned int p1 = 0xFFFFFFFF; unsigned int p2 = __float_as_uint(r.w * r.w); 
        optixTrace(params.handle, origin, direction, 0.0f, 1e-5f, 0.0f, 1u, OPTIX_RAY_FLAG_NONE, 0u, 1u, 0u, p0, p1, p2);
    "#;
    let anyhit_cuda = r#"
        unsigned int hit_primitive_id = optixGetPrimitiveIndex(); PayloadBundle* bundle = params.bundle;
        unsigned int __idx = atomicAdd((unsigned int*)bundle->dyn_lens[0], 1u);
        if (__idx < bundle->dyn_caps[0]) { ((unsigned int*)bundle->dyn_ptrs[0])[__idx] = hit_primitive_id; }
    "#;

    let expanded = quote! {
        {
            static __GPRT_TRACE_PIPELINE: std::sync::OnceLock<gprt_optix::OptixPipeline> = std::sync::OnceLock::new();
            let __pipeline: &gprt_optix::OptixPipeline = __GPRT_TRACE_PIPELINE.get_or_init(|| {
                let __ir = gprt_ir::RtProgram {
                    raygen_body: gprt_ir::ShaderNode::RawCuda(#raygen_cuda.to_string()),
                    anyhit_body: Some(gprt_ir::ShaderNode::RawCuda(#anyhit_cuda.to_string())),
                    miss_body: None, closesthit_body: None, intersection_body: None,
                    payload_layout: vec![], schedule: #index.schedule.clone(), array_indices: std::collections::HashMap::new(),
                };
                gprt_optix::OptixPipeline::new(&gprt_codegen::compile_program(&__ir), false)
            });
            let __num_rays: usize = #rays.len();
            __pipeline.register_array_batched("flat_out", 1000000u32, 1); 
            __pipeline.trace_scene(&mut #index.scene, &#rays, &#index.schedule);

            let __batched = __pipeline.retrieve_array_batched("flat_out", 1, 1000000);
            #array_ident.clear();
            let __len: usize = (__batched.1[0] as usize).min(1000000);
            for __i in 0..__len { #array_ident.push(__batched.0[__i]); }
        }
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn gprt_trace_batched(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as TraceBatchedInput);
    let index = input.index;
    let rays = input.rays;
    let cap = input.cap;

    let raygen_cuda = r#"
        uint3 launch_idx = optixGetLaunchIndex(); int idx = launch_idx.x; if (idx >= params.num_rays) return;
        
        PayloadBundle* bundle = params.bundle;
        ((unsigned int*)bundle->dyn_lens[0])[idx] = 0;

        float4 r = params.rays[idx]; float3 origin = make_float3(r.x, r.y, r.z); float3 direction = make_float3(1.0f, 0.0f, 0.0f);
        unsigned int p0 = __float_as_uint(r.w); unsigned int p1 = 0xFFFFFFFF; unsigned int p2 = __float_as_uint(r.w * r.w);
        optixTrace(params.handle, origin, direction, 0.0f, r.w, 0.0f, 1u, OPTIX_RAY_FLAG_NONE, 0u, 1u, 0u, p0, p1, p2);
    "#;
    
    let anyhit_cuda = r#"
        unsigned int prim_id = optixGetPrimitiveIndex();
        float dist_sq = __uint_as_float(optixGetAttribute_0()); 
        float worst_dist_sq = __uint_as_float(optixGetPayload_2());
        if (dist_sq > worst_dist_sq) { optixIgnoreIntersection(); return; }

        PayloadBundle* bundle = params.bundle; unsigned int __qid = optixGetLaunchIndex().x;
        unsigned int __idx = atomicAdd((unsigned int*)bundle->dyn_lens[0] + __qid, 1u);
        if (__idx < bundle->dyn_caps[0]) {
            ((unsigned int*)bundle->dyn_ptrs[0])[__qid * bundle->dyn_caps[0] + __idx] = prim_id;
        }
    "#;

    let expanded = quote! {
        {
            static __GPRT_BATCHED_PIPELINE: std::sync::OnceLock<gprt_optix::OptixPipeline> = std::sync::OnceLock::new();
            let __pipeline: &gprt_optix::OptixPipeline = __GPRT_BATCHED_PIPELINE.get_or_init(|| {
                let __ir = gprt_ir::RtProgram {
                    raygen_body: gprt_ir::ShaderNode::RawCuda(#raygen_cuda.to_string()),
                    anyhit_body: Some(gprt_ir::ShaderNode::RawCuda(#anyhit_cuda.to_string())),
                    miss_body: None, closesthit_body: None, intersection_body: None,
                    payload_layout: vec![], schedule: #index.schedule.clone(), array_indices: std::collections::HashMap::new(),
                };
                gprt_optix::OptixPipeline::new(&gprt_codegen::compile_program(&__ir), false)
            });
            let __num_rays: usize = #rays.len();
            let __cap: u32 = #cap;
            __pipeline.register_array_batched("batched_out", __cap, __num_rays);
            __pipeline.trace_scene(&mut #index.scene, &#rays, &#index.schedule);
            
            let __batched_data: (Vec<u32>, Vec<u32>) = __pipeline.retrieve_array_batched("batched_out", __num_rays, __cap as usize);
            (__batched_data.0, __batched_data.1, __cap)
        }
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn r_nn(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as RnnInput);
    let data = input.data;
    let queries = input.queries;
    let radius = input.radius;
    let output = input.output;

    let expanded = quote! {
        {
            let mut __gprt_builder = gprt_macros::gprt_build_index!(#data, |__p: &gprt_core::Vec3| gprt_core::Sphere { center: *__p, radius: #radius });           
            let mut __gprt_rays: Vec<gprt_core::Ray> = Vec::with_capacity(#queries.len());
            for __i in 0..#queries.len() { __gprt_rays.push(gprt_core::Ray::query(#queries[__i], #radius)); }
            
            gprt_macros::gprt_trace!(__gprt_builder, __gprt_rays, |hit| {
                #output.push(hit.primitive_id);
                gprt_core::HitAction::Continue
            });
        }
    };
    TokenStream::from(expanded)
}

#[proc_macro]
pub fn barnes_hut(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as BarnesHutInput);
    let bodies = input.bodies;
    let theta = input.theta;
    let _g_const = input.g_const;
    let output = input.output;

    let bh_raygen = r#"
        unsigned int idx = optixGetLaunchIndex().x; if (idx >= params.num_bodies) return;
        unsigned int p0 = 0; unsigned int p1 = 0, p2 = 0, p3 = 0;
        while (p0 < params.num_nodes) {
            NodeSpatial spatial = params.tree_spatial[p0];
            float3 diff = make_float3(spatial.com.x - params.bodies[idx].pos.x, spatial.com.y - params.bodies[idx].pos.y, spatial.com.z - params.bodies[idx].pos.z);
            float dist_sq = gprt_dot(diff, diff); float d = sqrt(dist_sq);
            float tmax = fmaxf(1e-6f, params.theta * d) + 1e-5f;
            float3 ray_origin = make_float3(0.0f, (float)p0 * 3.0f - 0.1f, -0.1f); float3 ray_dir = make_float3(1.0f, 0.0f, 0.0f);
            optixTrace(params.handle, ray_origin, ray_dir, 0.0f, tmax, 0.0f, 255u, OPTIX_RAY_FLAG_NONE, 0u, 4u, 0u, p0, p1, p2, p3);
        }
        params.out_forces[idx] = make_float3(__uint_as_float(p1), __uint_as_float(p2), __uint_as_float(p3));
    "#;
    let bh_closesthit = r#"
        unsigned int current_node = optixGetPayload_0();
        float3 state = make_float3(__uint_as_float(optixGetPayload_1()), __uint_as_float(optixGetPayload_2()), __uint_as_float(optixGetPayload_3()));
        unsigned int qid = optixGetLaunchIndex().x;
        NodeSpatial spatial = params.tree_spatial[current_node]; NodeRouting routing = params.tree_routing[current_node];
        float3 diff = make_float3(spatial.com.x - params.bodies[qid].pos.x, spatial.com.y - params.bodies[qid].pos.y, spatial.com.z - params.bodies[qid].pos.z);
        float dist_sq = gprt_dot(diff, diff);
        if (routing.is_leaf == 1) {
            for (unsigned int i = 0; i < routing.num_particles; i = i + 1) {
                Body b = params.bodies[routing.bucket_start + i];
                float3 d_diff = make_float3(b.pos.x - params.bodies[qid].pos.x, b.pos.y - params.bodies[qid].pos.y, b.pos.z - params.bodies[qid].pos.z);
                float d_dist_sq = gprt_dot(d_diff, d_diff);
                if (d_dist_sq > 1e-10f) { float denom = d_dist_sq * sqrt(d_dist_sq + 1e-6f); float force_mag = (6.674e-11f * params.bodies[qid].mass * b.mass) / denom; state.x += d_diff.x * force_mag; state.y += d_diff.y * force_mag; state.z += d_diff.z * force_mag; }
            } optixSetPayload_0(routing.autorope);
        } else {
            if (dist_sq > 1e-10f) { float denom = dist_sq * sqrt(dist_sq + 1e-6f); float force_mag = (6.674e-11f * params.bodies[qid].mass * spatial.mass) / denom; state.x += diff.x * force_mag; state.y += diff.y * force_mag; state.z += diff.z * force_mag; }
            optixSetPayload_0(routing.autorope);
        }
        optixSetPayload_1(__float_as_uint(state.x)); optixSetPayload_2(__float_as_uint(state.y)); optixSetPayload_3(__float_as_uint(state.z));
    "#;
    let bh_miss = r#"
        unsigned int current_node = optixGetPayload_0();
        float3 state = make_float3(__uint_as_float(optixGetPayload_1()), __uint_as_float(optixGetPayload_2()), __uint_as_float(optixGetPayload_3()));
        unsigned int qid = optixGetLaunchIndex().x; NodeRouting routing = params.tree_routing[current_node];
        if (routing.is_leaf == 1) {
            for (unsigned int i = 0; i < routing.num_particles; i = i + 1) {
                Body b = params.bodies[routing.bucket_start + i];
                float3 d_diff = make_float3(b.pos.x - params.bodies[qid].pos.x, b.pos.y - params.bodies[qid].pos.y, b.pos.z - params.bodies[qid].pos.z);
                float d_dist_sq = gprt_dot(d_diff, d_diff);
                if (d_dist_sq > 1e-10f) { float denom = d_dist_sq * sqrt(d_dist_sq + 1e-6f); float force_mag = (6.674e-11f * params.bodies[qid].mass * b.mass) / denom; state.x += d_diff.x * force_mag; state.y += d_diff.y * force_mag; state.z += d_diff.z * force_mag; }
            } optixSetPayload_0(routing.next_idx);
        } else { optixSetPayload_0(routing.next_idx); }
        optixSetPayload_1(__float_as_uint(state.x)); optixSetPayload_2(__float_as_uint(state.y)); optixSetPayload_3(__float_as_uint(state.z));
    "#;

    let expanded = quote! {
        {
            let __tree = gprt_core::BarnesHutTree::build(&#bodies);
            let __num_nodes = __tree.nodes.len();
            let mut __h_spatial = Vec::with_capacity(__num_nodes);
            let mut __h_routing = Vec::with_capacity(__num_nodes);
            for node in &__tree.nodes {
                __h_spatial.push(gprt_core::NodeSpatial { com: node.com, mass: node.mass });
                __h_routing.push(gprt_core::NodeRouting { next_idx: node.next_idx, autorope: node.autorope, is_leaf: node.is_leaf, width: node.width, bucket_start: node.bucket_start, num_particles: node.num_particles, _pad: [0, 0] });
            }
            let __schedule = gprt_ir::Schedule::default();
            static __GPRT_BH_PIPELINE: std::sync::OnceLock<gprt_optix::OptixPipeline> = std::sync::OnceLock::new();
            let __pipeline: &gprt_optix::OptixPipeline = __GPRT_BH_PIPELINE.get_or_init(|| {
                let __ir = gprt_ir::RtProgram {
                    raygen_body: gprt_ir::ShaderNode::RawCuda(#bh_raygen.to_string()), anyhit_body: None,
                    miss_body: Some(gprt_ir::ShaderNode::RawCuda(#bh_miss.to_string())),
                    closesthit_body: Some(gprt_ir::ShaderNode::RawCuda(#bh_closesthit.to_string())),
                    intersection_body: None, payload_layout: vec![], schedule: __schedule, array_indices: std::collections::HashMap::new(),
                };
                gprt_optix::OptixPipeline::new(&gprt_codegen::compile_program(&__ir), true)
            });
            let __forces = __pipeline.execute_autorope_soa(&#bodies, &__h_spatial, &__h_routing, #theta as f32);
            #output.clear(); #output.extend(__forces);
        }
    };
    TokenStream::from(expanded)
}
