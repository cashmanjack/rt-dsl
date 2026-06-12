use gprt_core::Vec3;
use gprt_macros::k_nn;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn load_points(path: &str) -> Vec<Vec3> {
    let file = File::open(path).expect("Failed to open data file");
    let reader = BufReader::new(file);
    let mut points = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        if line.trim().is_empty() { continue; }

	let parts: Vec<f32> = line.split(|c: char| c == ',' || c.is_whitespace())
	    .filter(|s| !s.trim().is_empty())
	    .filter_map(|s| s.trim().parse().ok())
	    .collect();

        if parts.len() >= 3 {
            points.push(Vec3::new(parts[0], parts[1], parts[2]));
        } else if parts.len() == 2 {
            points.push(Vec3::new(parts[0], parts[1], 0.0));
        }
    }
    points
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <data.csv> <queries.csv> <k>", args[0]);
        std::process::exit(1);
    }

    let data_path = &args[1];
    let query_path = &args[2];
    let k: usize = args[3].parse().expect("K must be an integer");

    let dataset = load_points(data_path);
    let queries = load_points(query_path);
    
    // Warmup / Pipeline caching (OptiX caches PTX on first run)
    let mut warmup_out: Vec<u32> = Vec::new();
    let warmup_queries = vec![queries[0]];
    k_nn!(dataset, warmup_queries, k, warmup_out);

    // Actual Benchmark
    let start = Instant::now();
    let mut k_neighbors: Vec<u32> = Vec::new();
    k_nn!(dataset, queries, k, k_neighbors);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Machine-readable output: Dataset, NumQueries, K, Time_ms, Total_Neighbors_Found
    println!("{},{},{},{:.3},{}", data_path, queries.len(), k, elapsed_ms, k_neighbors.len());
}
