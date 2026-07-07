use gprt_core::{Vec3, Body};
use gprt_macros::barnes_hut;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn load_galaxy(path: &str) -> Vec<Body> {
    let file = File::open(path).expect("Failed to open dataset");
    let reader = BufReader::new(file);
    let mut bodies = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        if line.trim().is_empty() { continue; }
        
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 4 {
            let x: f32 = parts[0].trim().parse().unwrap_or(0.0);
            let y: f32 = parts[1].trim().parse().unwrap_or(0.0);
            let z: f32 = parts[2].trim().parse().unwrap_or(0.0);
            let mass: f32 = parts[3].trim().parse().unwrap_or(0.0);
            
            // Skip the 0,0,0,0 dummy line
            if mass <= 0.0 { continue; }

            bodies.push(Body {
                pos: Vec3::new(x, y, z),
                mass,
                velocity: Vec3::new(0.0, 0.0, 0.0),
            });
        }
    }
    bodies
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <dataset.txt> <theta>", args[0]);
        std::process::exit(1);
    }

    let dataset_path = &args[1];
    let theta: f32 = args[2].parse().expect("Theta must be a float");

    println!("Loading Galaxy dataset from {}...", dataset_path);
    let bodies = load_galaxy(dataset_path);
    println!("Loaded {} bodies.", bodies.len());

    let mut output_forces = Vec::new();

    println!("Compiling and launching RT-BarnesHut via GPRT DSL...");
    let start = Instant::now();

    // THE DECOUPLED DSL INVOCATION
    barnes_hut!(bodies, theta, 6.674e-11, output_forces);

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    let total_force_mag: f32 = output_forces.iter()
        .map(|f| (f.x*f.x + f.y*f.y + f.z*f.z).sqrt())
        .sum();
        
    println!("Total Force Magnitude: {:.4e}", total_force_mag);

    println!("\n=== GPRT DSL BARNES-HUT REPORT ===");
    println!("Execution Time : {:.3} ms", elapsed_ms);
    println!("Forces Computed: {}", output_forces.len());
    println!("==================================\n");
}
