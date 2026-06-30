mod io;

use gprt_macros::k_nn;
use gprt_ir::{Schedule, RadiusHeuristic};
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <data.csv> <queries.csv> <k> [percentile] [mult] [dump]", args[0]);
        std::process::exit(1);
    }

    let data_path = &args[1];
    let query_path = &args[2];
    let k: usize = args[3].parse().expect("K must be an integer");
    
    let mut percentile: f32 = 0.10;
    let mut mult: f32 = 3.0;
    let mut dump_results = false;

    let mut float_idx = 0;
    for arg in args.iter().skip(4) {
        if arg == "dump" {
            dump_results = true;
        } else if let Ok(val) = arg.parse::<f32>() {
            if float_idx == 0 { percentile = val; float_idx += 1; } 
            else if float_idx == 1 { mult = val; float_idx += 1; }
        }
    }

    println!("[TUNING] Percentile: {:.2}, Multiplier: {:.1}", percentile, mult);

    let t0 = Instant::now();
    let dataset = io::load_points_fast(data_path);
    let queries = io::load_points_fast(query_path);
    println!("[IO] Loaded {} points and {} queries in {:.2}ms", 
             dataset.len(), queries.len(), t0.elapsed().as_secs_f32() * 1000.0);

    let mut custom_schedule = Schedule::default();
    custom_schedule.radius_heuristic = RadiusHeuristic::SampledPercentile(percentile);
    custom_schedule.radius_increment_mult = mult;

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
