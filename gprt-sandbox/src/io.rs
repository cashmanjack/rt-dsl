use gprt_core::Vec3;
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;

/// Blazing fast, multi-threaded CSV parser using memory mapping and Rayon.
/// Parses 10 Million points in < 1 second.
pub fn load_points_fast(path: &str) -> Vec<Vec3> {
    let file = File::open(path).expect("Failed to open dataset");
    
    // Memory map the file directly into RAM (Zero-copy OS level mapping)
    let mmap = unsafe { Mmap::map(&file).expect("Failed to mmap") };
    
    // Split by newlines and parse in parallel across all CPU cores
    mmap.par_split(|&b| b == b'\n')
        .filter_map(|line| {
            if line.is_empty() { return None; }
            
            // Fast UTF-8 validation and splitting
            let s = std::str::from_utf8(line).ok()?;
            let mut parts = s.split(',');
            
            let x = parts.next()?.trim().parse::<f32>().ok()?;
            let y = parts.next()?.trim().parse::<f32>().ok()?;
            let z = parts.next()?.trim().parse::<f32>().ok()?;
            
            Some(Vec3::new(x, y, z))
        })
        .collect() // Rayon automatically builds the Vec in parallel
}
