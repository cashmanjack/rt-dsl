mod io;

use gprt_macros::{k_nn, gprt_autotune};
use gprt_ir::{Schedule, RadiusHeuristic};
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <data.csv> <queries.csv> <k> [percentile] [mult] [dump] [tune]", args[0]);
        std::process::exit(1);
    }

    let data_path = &args[1];
    let query_path = &args[2];
    let k: usize = args[3].parse().expect("K must be an integer");
    
    let mut percentile: f32 = 0.10;
    let mut mult: f32 = 3.0;
    let mut dump_results = false;
    let mut tune_mode = false;

    let mut float_idx = 0;
    for arg in args.iter().skip(4) {
        match arg.as_str() {
            "dump" => dump_results = true,
            "tune" => tune_mode = true,
            _ => {
                if let Ok(val) = arg.parse::<f32>() {
                    if float_idx == 0 { percentile = val; float_idx += 1; } 
                    else if float_idx == 1 { mult = val; float_idx += 1; }
                }
            }
        }
    }

    // ==========================================
    // 1. LOAD DATA FIRST
    // ==========================================
    let t0 = Instant::now();
    let dataset = io::load_points_fast(data_path);
    let queries = io::load_points_fast(query_path);
    println!("[IO] Loaded {} points and {} queries in {:.2}ms", 
             dataset.len(), queries.len(), t0.elapsed().as_secs_f32() * 1000.0);

    // ==========================================
    // 2. SCHEDULE SELECTION (Tune or Load)
    // ==========================================

    let custom_schedule = if tune_mode {
        println!("[TUNE] Starting Native Full-Dataset Autotuner...");
        let tuned = gprt_autotune!(dataset, queries, k);
        if let Ok(json) = serde_json::to_string_pretty(&tuned) {
            let _ = std::fs::write("gprt_wisdom.json", json);
            println!("[TUNE] Saved optimal schedule to gprt_wisdom.json");
        }
        tuned
    } else {
        let mut schedule = Schedule::default();
        if let Ok(json) = std::fs::read_to_string("gprt_wisdom.json") {
            if let Ok(wisdom) = serde_json::from_str::<Schedule>(&json) {
                println!("[LOAD] Loaded auto-tuned schedule from gprt_wisdom.json");
                schedule = wisdom;
            } else {
                schedule.radius_heuristic = RadiusHeuristic::SampledPercentile(percentile);
                schedule.radius_increment_mult = mult;
            }
        } else {
            println!("[LOAD] Using CLI defaults: P={:.2}, M={:.1}", percentile, mult);
            schedule.radius_heuristic = RadiusHeuristic::SampledPercentile(percentile);
            schedule.radius_increment_mult = mult;
        }
        schedule
    };

    // ==========================================
    // 3. EXECUTION TIMER (Strictly measures k_nn!)
    // ==========================================
    let t1 = Instant::now();
    let mut knn_neighbors: Vec<Vec<u32>> = Vec::new();
    k_nn!(dataset, queries, k, knn_neighbors, custom_schedule);
    let t_total = t1.elapsed().as_secs_f64() * 1000.0;
    
    let total_neighbors: usize = knn_neighbors.iter().map(|v| v.len()).sum();

    if dump_results {
        let mut f = File::create("knn_results.txt").unwrap();
        for (i, neighbors) in knn_neighbors.iter().enumerate() {
            writeln!(f, "Q{}:{:?}", i, neighbors).unwrap();
        }
    }

    println!("{},{},{},{:.3},{}", data_path, queries.len(), k, t_total, total_neighbors);
}
