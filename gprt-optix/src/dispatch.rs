use gprt_core::{Vec3, Scene, Sphere, Ray};
use gprt_ir::{Schedule, RadiusHeuristic, ShaderNode, RtProgram};
use gprt_codegen::compile_program;
use crate::{OptixPipeline, IndexBuilder};
use std::time::Instant;
use std::fs;
use rayon::prelude::*;

fn load_wisdom_or_use_provided(provided: &Schedule) -> Schedule {
    if let Ok(json) = fs::read_to_string("gprt_wisdom.json") {
        if let Ok(wisdom) = serde_json::from_str::<Schedule>(&json) {
            println!("[GPRT] Loaded auto-tuned schedule from gprt_wisdom.json (Delete this file to use CLI args)");
            return wisdom;
        }
    }
    provided.clone()
}

pub fn execute_knn(
    data: &[Vec3],
    queries: &[Vec3],
    k: usize,
    schedule: &Schedule,
    out: &mut Vec<Vec<u32>>,
) {
    execute_knn_with_wisdom(data, queries, k, schedule, out, true);
}

pub fn execute_knn_with_wisdom(
    data: &[Vec3],
    queries: &[Vec3],
    k: usize,
    schedule: &Schedule,
    out: &mut Vec<Vec<u32>>,
    use_wisdom: bool,
) -> f64 {
    let final_schedule = if use_wisdom {
        load_wisdom_or_use_provided(schedule)
    } else {
        schedule.clone()
    };

    let n_data = data.len();
    let n_queries = queries.len();

    let sorted_data: Vec<Vec3>;
    let data_ref: &[Vec3];

    if final_schedule.use_morton_lbv {
        let t0 = std::time::Instant::now();
        let sorted_indices = gprt_core::morton::sort_by_morton(data);
        sorted_data = sorted_indices.iter().map(|&i| data[i]).collect();
        data_ref = &sorted_data;
    } else {
        data_ref = data;
    }

    let mut min_b = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max_b = Vec3::new(f32::MIN, f32::MIN, f32::MIN);
    for p in data_ref {
        if p.x < min_b.x { min_b.x = p.x; } if p.y < min_b.y { min_b.y = p.y; } if p.z < min_b.z { min_b.z = p.z; }
        if p.x > max_b.x { max_b.x = p.x; } if p.y > max_b.y { max_b.y = p.y; } if p.z > max_b.z { max_b.z = p.z; }
    }
    
    let mut seed: u64 = 0x123456789ABCDEF;
    let mut next_rand = || -> usize {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as usize
    };
    let sample_size = if 5000 < n_data { 5000 } else { n_data };
    let query_sample_size = if 100 < n_queries { 100 } else { n_queries };
    let mut sampled_kth_dists: Vec<f32> = Vec::with_capacity(query_sample_size);

    for _ in 0..query_sample_size {
        let qi = next_rand() % n_queries;
        let q = queries[qi];
        let mut dists: [f32; 4] = [f32::MAX; 4]; 
        for _ in 0..sample_size {
            let pi = next_rand() % n_data;
            let p = data_ref[pi];
            let d = ((q.x-p.x).powi(2) + (q.y-p.y).powi(2) + (q.z-p.z).powi(2)).sqrt();
            if d < dists[3] {
                dists[3] = d;
                if dists[3] < dists[2] { dists.swap(2, 3); }
                if dists[2] < dists[1] { dists.swap(1, 2); }
                if dists[1] < dists[0] { dists.swap(0, 1); }
            }
        }
        if dists[3] != f32::MAX { sampled_kth_dists.push(dists[3]); }
    }
    
    let mut current_radius: f32 = 10.0;
    if !sampled_kth_dists.is_empty() {
        sampled_kth_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        match final_schedule.radius_heuristic {
            RadiusHeuristic::SampledMax => current_radius = *sampled_kth_dists.last().unwrap() * 1.1,
            RadiusHeuristic::SampledPercentile(p) => {
                let idx = ((sampled_kth_dists.len() as f32) * p).min((sampled_kth_dists.len() - 1) as f32) as usize;
                current_radius = sampled_kth_dists[idx] * 1.1;
            }
            RadiusHeuristic::Fixed(r) => current_radius = r,
        }
    }

    let scene = Scene::build(data_ref.iter().map(|p| Sphere { center: *p, radius: current_radius }));
    let mut builder = IndexBuilder { scene, schedule: final_schedule.clone() };

    let raygen_cuda = r#"
        uint3 launch_idx = optixGetLaunchIndex(); int idx = launch_idx.x; if (idx >= params.num_rays) return;
        float4 r = params.rays[idx]; float3 origin = make_float3(r.x, r.y, r.z); float3 direction = make_float3(1.0f, 0.0f, 0.0f);
        
        // FIXED: 1e-3f clears FP32 precision culls while remaining a Point Ray
        float tmax = 1e-3f; 
        unsigned int p0 = 0; 
        unsigned int p1 = 0xFFFFFFFF; 
        unsigned int p2 = 0; 
        unsigned int p3 = __float_as_uint(r.w * r.w); 
        optixTrace(params.handle, origin, direction, 0.0f, tmax, 0.0f, 1u, OPTIX_RAY_FLAG_NONE, 0u, 1u, 0u, p0, p1, p2, p3);
    "#;
    
    // FIXED: Pure AnyHit. No early returns, no self-intersection logic. Just record and keep alive.
    let anyhit_cuda = r#"
        unsigned int prim_id = optixGetPrimitiveIndex();
        PayloadBundle* bundle = params.bundle; unsigned int __qid = optixGetLaunchIndex().x;
        unsigned int __idx = atomicAdd((unsigned int*)bundle->dyn_lens[0] + __qid, 1u);
        if (__idx < bundle->dyn_caps[0]) { ((unsigned int*)bundle->dyn_ptrs[0])[__qid * bundle->dyn_caps[0] + __idx] = prim_id; }
        optixIgnoreIntersection();
    "#;

    let ir = RtProgram {
        raygen_body: ShaderNode::RawCuda(raygen_cuda.to_string()),
        anyhit_body: Some(ShaderNode::RawCuda(anyhit_cuda.to_string())),
        miss_body: None, closesthit_body: None, intersection_body: None,
        payload_layout: vec![], schedule: final_schedule.clone(), array_indices: std::collections::HashMap::new(),
    };
    let pipeline = OptixPipeline::new(&compile_program(&ir), false);

    let requested_cap = final_schedule.max_hits_per_query;
    let max_allowed_cap = (250_000_000 / n_queries.max(1)) as u32;
    let safe_cap = requested_cap.min(max_allowed_cap).min(n_data as u32);
    let dynamic_cap: u32 = safe_cap;

    pipeline.register_array_batched("batched_out", dynamic_cap, n_queries);

    let mut active_indices: Vec<usize> = (0..n_queries).collect();
    let mut per_query_results: Vec<Vec<u32>> = vec![Vec::new(); n_queries];
    let mut total_gpu_search_ms: f64 = 0.0;
    let mut iteration: usize = 0;
    let mut total_saturated = 0usize;
    let mut total_queries_processed = 0usize;

    loop {
        if active_indices.is_empty() || iteration > 30 { break; }
        iteration += 1;
        let num_active = active_indices.len();

        let gprt_rays: Vec<Ray> = active_indices.iter().map(|&qi| Ray::query(queries[qi], current_radius)).collect();
        let t0 = Instant::now();
        
        pipeline.trace_scene(&mut builder.scene, &gprt_rays, &final_schedule);
        total_gpu_search_ms += t0.elapsed().as_secs_f64() * 1000.0;
        
        let (flat_results, lengths) = pipeline.retrieve_array_batched("batched_out", num_active, dynamic_cap as usize);
        
        let (next_active_vec, results_updates, saturated_count): (Vec<Option<usize>>, Vec<(usize, Vec<u32>)>, usize) = active_indices.par_iter().enumerate().map(|(local_idx, &global_qi)| {
            let hit_count = (lengths[local_idx] as usize).min(dynamic_cap as usize);
            let is_saturated = lengths[local_idx] >= dynamic_cap;
            let start_idx = local_idx * (dynamic_cap as usize);
            let q_pos = queries[global_qi];
            
            let mut hits_with_dist: Vec<(u32, f32)> = Vec::with_capacity(hit_count);
            for i in 0..hit_count {
                let id = flat_results[start_idx + i];
                let p = data_ref[id as usize];
                let dx = q_pos.x - p.x; let dy = q_pos.y - p.y; let dz = q_pos.z - p.z;
                let d2 = dx*dx + dy*dy + dz*dz;
                
                // FIXED: Filter self-intersection on the CPU to keep the AnyHit shader perfectly clean
                if d2 > 1e-10f32 { 
                    hits_with_dist.push((id, d2));
                }
            }
            
            hits_with_dist.sort_by(|a: &(u32, f32), b: &(u32, f32)| a.1.partial_cmp(&b.1).unwrap());
            hits_with_dist.dedup_by(|a, b| a.0 == b.0);

            let mut res = Vec::new();
            let mut is_active = false;

            if !is_saturated && hits_with_dist.len() >= k {
                for i in 0..k { res.push(hits_with_dist[i].0); }
            } else if is_saturated || iteration > 15 {
                let take = if hits_with_dist.len() > k { k } else { hits_with_dist.len() };
                for i in 0..take { res.push(hits_with_dist[i].0); }
            } else {
                is_active = true;
            }
            
            (if is_active { Some(global_qi) } else { None }, (global_qi, res), if is_saturated { 1 } else { 0 })
        }).fold(
            || (Vec::new(), Vec::new(), 0usize),
            |mut acc, (active, result, saturated)| {
                acc.0.push(active);
                acc.1.push(result);
                acc.2 += saturated;
                acc
            }
        ).reduce(
            || (Vec::new(), Vec::new(), 0usize),
            |mut a, b| {
                a.0.extend(b.0);
                a.1.extend(b.1);
                a.2 += b.2;
                a
            }
        );
        
        total_saturated += saturated_count;
        total_queries_processed += active_indices.len();
        
        active_indices = next_active_vec.into_iter().flatten().collect();
        for (qi, res) in results_updates {
            if !res.is_empty() { per_query_results[qi] = res; }
        }

        if active_indices.is_empty() { break; }

        current_radius *= final_schedule.radius_increment_mult;
        for prim in &mut builder.scene.primitives { prim.radius = current_radius; }
        builder.scene.mark_dirty();
    }
    
    let saturation_ratio = if total_queries_processed > 0 {
        total_saturated as f64 / total_queries_processed as f64
    } else {
        0.0
    };
    
    println!("[GPRT_STATS] search_ms={:.3}, saturation={:.2}%", total_gpu_search_ms, saturation_ratio * 100.0);
    
    *out = per_query_results;
    pipeline.clear_bvh_cache();
    
    saturation_ratio
}
