use gprt_core::Vec3;
use gprt_ir::{Schedule, RadiusHeuristic};
use gprt_optix::dispatch::execute_knn_with_wisdom;
use memmap2::Mmap;
use rayon::prelude::*;
use std::{fs::File, env, time::Instant};
use std::io::{self, Write};

fn load_points_fast(path: &str) -> Vec<Vec3> {
    let file = File::open(path).expect("Failed to open dataset");
    let mmap = unsafe { Mmap::map(&file).expect("Failed to mmap") };
    mmap.par_split(|&b| b == b'\n').filter_map(|line| {
        if line.is_empty() { return None; }
        let s = std::str::from_utf8(line).ok()?;
        let mut parts = s.split(',');
        let x = parts.next()?.trim().parse::<f32>().ok()?;
        let y = parts.next()?.trim().parse::<f32>().ok()?;
        let z = parts.next()?.trim().parse::<f32>().ok()?;
        Some(Vec3::new(x, y, z))
    }).collect()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: gprt-profiler <data.csv> <queries.csv> <k>");
        return;
    }
    
    println!("Loading data...");
    let data = load_points_fast(&args[1]);
    let queries = load_points_fast(&args[2]);
    let k: usize = args[3].parse().unwrap();
    
    println!("Profiling {} points and {} queries...", data.len(), queries.len());
    
    // 1. SMARTER SEARCH SPACE
    // Removed 0.50 and 0.90. TrueKNN almost never benefits from massive starting radii 
    // because it causes catastrophic AABB overlap on Iteration 1.
    let percentiles = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20];
    let multipliers = [2.0, 3.0, 4.0];
    let caps = [500, 2000, 5000, 10_000];
    
    // Generate all possible configurations
    let mut all_configs = Vec::new();
    for &p in &percentiles {
        for &m in &multipliers {
            for &cap in &caps {
                let mut s = Schedule::default();
                s.radius_heuristic = RadiusHeuristic::SampledPercentile(p);
                s.radius_increment_mult = m;
                s.max_hits_per_query = cap;
                all_configs.push(s);
            }
        }
    }
    
    // 2. THE FUNNEL (Progressive Subsampling)
    // (Target subset size, Number of top configs to keep for the next stage)
    let funnel = [
        (5_000, 20),   // Stage 1: Test all 63 configs on 5k points. Keep top 20.
        (25_000, 5),   // Stage 2: Test top 20 on 25k points. Keep top 5.
        (100_000, 1),  // Stage 3: Test top 5 on 100k points. Keep the absolute winner.
    ];
    
    let mut current_candidates = all_configs;
    let mut best_overall_schedule = Schedule::default();
    let mut best_overall_time = f64::MAX;
    
    for (stage_idx, (target_size, keep_top_n)) in funnel.iter().enumerate() {
        let actual_size = (*target_size).min(data.len()).min(queries.len());
        let sub_data = &data[..actual_size];
        let sub_queries = &queries[..actual_size];
        
        println!("\n=== Funnel Stage {}: Testing {} configs on {} points ===", 
                 stage_idx + 1, current_candidates.len(), actual_size);
        
        let mut stage_results: Vec<(Schedule, f64)> = Vec::new();
        
        for (i, sched) in current_candidates.iter().enumerate() {
            let p_val = match sched.radius_heuristic { RadiusHeuristic::SampledPercentile(p) => p, _ => 0.0 };
            print!("\r  [{}/{}] P={:.3}, M={:.1}, cap={}...   ", 
                   i + 1, current_candidates.len(), p_val, sched.radius_increment_mult, sched.max_hits_per_query);
            io::stdout().flush().unwrap();
            
            let mut out = Vec::new();
            let t0 = Instant::now();
            let saturation = execute_knn_with_wisdom(sub_data, sub_queries, k, sched, &mut out, false);
            let elapsed = t0.elapsed().as_secs_f64();
            
            // Reject configurations that saturate (they will fail accuracy validation)
            if saturation > 0.05 {
                continue;
            }
            
            stage_results.push((sched.clone(), elapsed));
        }
        println!(); // Newline after progress bar
        
        // Sort by execution time
        stage_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        if let Some((best_sched, best_time)) = stage_results.first() {
            println!("  -> Stage Winner: P={:?}, M={:.1}, cap={} ({:.3}s)", 
                     best_sched.radius_heuristic, best_sched.radius_increment_mult, best_sched.max_hits_per_query, best_time);
            
            // Track the absolute best from the largest subset tested
            if actual_size >= 25_000 && *best_time < best_overall_time {
                best_overall_time = *best_time;
                best_overall_schedule = best_sched.clone();
            }
        }
        
        // Keep the top N configs for the next funnel stage
        let keep_count = (*keep_top_n).min(stage_results.len());
        current_candidates = stage_results.into_iter().take(keep_count).map(|(s, _)| s).collect();
        
        if current_candidates.is_empty() {
            println!("WARNING: All configs saturated or failed! Falling back to defaults.");
            break;
        }
    }
    
    // Save the winner
    let json = serde_json::to_string_pretty(&best_overall_schedule).unwrap();
    std::fs::write("gprt_wisdom.json", json).unwrap();
    
    println!("\n=== FINAL AUTO-TUNED RESULT ===");
    println!("Best Schedule: P={:?}, M={:.1}, cap={}", 
             best_overall_schedule.radius_heuristic, 
             best_overall_schedule.radius_increment_mult,
             best_overall_schedule.max_hits_per_query);
    println!("Saved to gprt_wisdom.json");
}
