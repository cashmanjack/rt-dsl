use crate::Vec3;

// Expand 10-bit integer to 30-bit by inserting 2 zeros between each bit
fn expand_bits(x: u32) -> u32 {
    let mut x = x & 0x000003ff;  // Keep only 10 bits
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x <<  8)) & 0x0300F00F;
    x = (x | (x <<  4)) & 0x030C30C3;
    x = (x | (x <<  2)) & 0x09249249;
    x
}

// Compute 30-bit Morton code for 3D point
pub fn morton(x: u32, y: u32, z: u32) -> u32 {
    (expand_bits(x) << 2) | (expand_bits(y) << 1) | expand_bits(z)
}

// Normalize 3D points to [0, 1023] range and compute Morton codes
pub fn compute_morton_codes(points: &[Vec3]) -> Vec<u32> {
    if points.is_empty() {
        return Vec::new();
    }
    
    // Find bounding box
    let mut min_b = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max_b = Vec3::new(f32::MIN, f32::MIN, f32::MIN);
    
    for p in points {
        min_b.x = min_b.x.min(p.x);
        min_b.y = min_b.y.min(p.y);
        min_b.z = min_b.z.min(p.z);
        max_b.x = max_b.x.max(p.x);
        max_b.y = max_b.y.max(p.y);
        max_b.z = max_b.z.max(p.z);
    }
    
    let extent_x = (max_b.x - min_b.x).max(1e-6);
    let extent_y = (max_b.y - min_b.y).max(1e-6);
    let extent_z = (max_b.z - min_b.z).max(1e-6);
    
    // Compute Morton codes
    points.iter().map(|p| {
        let x = ((p.x - min_b.x) / extent_x * 1023.0) as u32;
        let y = ((p.y - min_b.y) / extent_y * 1023.0) as u32;
        let z = ((p.z - min_b.z) / extent_z * 1023.0) as u32;
        morton(x.min(1023), y.min(1023), z.min(1023))
    }).collect()
}

// Sort points by Morton code and return sorted indices
pub fn sort_by_morton(points: &[Vec3]) -> Vec<usize> {
    let morton_codes = compute_morton_codes(points);
    let mut indices: Vec<usize> = (0..points.len()).collect();
    indices.sort_by_key(|&i| morton_codes[i]);
    indices
}
