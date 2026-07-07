use std::ops::{Add, Sub, Mul, Div};
use std::sync::atomic::{AtomicU64, Ordering};

pub mod morton;

static NEXT_SCENE_ID: AtomicU64 = AtomicU64::new(1);

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    pub fn length(&self) -> f32 { (self.x * self.x + self.y * self.y + self.z * self.z).sqrt() }
    pub fn normalize(&self) -> Self { let len = self.length(); if len > 0.0 { *self / len } else { *self } }
    pub fn distance(&self, other: &Self) -> f32 { let d = *self - *other; d.length() }
    pub fn min(&self, other: &Self) -> Self { Self::new(self.x.min(other.x), self.y.min(other.y), self.z.min(other.z)) }
    pub fn max(&self, other: &Self) -> Self { Self::new(self.x.max(other.x), self.y.max(other.y), self.z.max(other.z)) }
    pub fn dot(&self, other: &Self) -> f32 { self.x * other.x + self.y * other.y + self.z * other.z }
    pub fn cross(&self, other: &Self) -> Self {
        Self::new(self.y * other.z - self.z * other.y, self.z * other.x - self.x * other.z, self.x * other.y - self.y * other.x)
    }
}

impl Add for Vec3 { type Output = Self; fn add(self, rhs: Self) -> Self { Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z) } }
impl Sub for Vec3 { type Output = Self; fn sub(self, rhs: Self) -> Self { Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z) } }
impl Mul<f32> for Vec3 { type Output = Self; fn mul(self, rhs: f32) -> Self { Self::new(self.x * rhs, self.y * rhs, self.z * rhs) } }
impl Div<f32> for Vec3 { type Output = Self; fn div(self, rhs: f32) -> Self { Self::new(self.x / rhs, self.y / rhs, self.z / rhs) } }

#[derive(Debug, Clone, Copy)]
pub struct AABB { pub min: Vec3, pub max: Vec3 }
impl AABB {
    pub fn empty() -> Self { Self { min: Vec3::new(f32::MAX, f32::MAX, f32::MAX), max: Vec3::new(f32::MIN, f32::MIN, f32::MIN) } }
    pub fn grow(&mut self, p: Vec3) { self.min = self.min.min(&p); self.max = self.max.max(&p); }
    pub fn merge(&mut self, other: &AABB) { self.min = self.min.min(&other.min); self.max = self.max.max(&other.max); }
    pub fn center(&self) -> Vec3 { (self.min + self.max) * 0.5 }
    pub fn intersect(&self, ray: &Ray) -> bool {
        let mut tmin = ray.tmin; let mut tmax = ray.tmax;
        let inv_d = Vec3::new(1.0 / ray.direction.x, 1.0 / ray.direction.y, 1.0 / ray.direction.z);
        let tx1 = (self.min.x - ray.origin.x) * inv_d.x; let tx2 = (self.max.x - ray.origin.x) * inv_d.x;
        tmin = tmin.max(tx1.min(tx2)); tmax = tmax.min(tx1.max(tx2));
        let ty1 = (self.min.y - ray.origin.y) * inv_d.y; let ty2 = (self.max.y - ray.origin.y) * inv_d.y;
        tmin = tmin.max(ty1.min(ty2)); tmax = tmax.min(ty1.max(ty2));
        let tz1 = (self.min.z - ray.origin.z) * inv_d.z; let tz2 = (self.max.z - ray.origin.z) * inv_d.z;
        tmin = tmin.max(tz1.min(tz2)); tmax = tmax.min(tz1.max(tz2));
        tmax >= tmin && tmax > 0.0
    }
}

#[derive(Debug, Clone)]
pub struct Ray { pub origin: Vec3, pub direction: Vec3, pub tmin: f32, pub tmax: f32 } 
impl Ray { 
    pub fn new(origin: Vec3, direction: Vec3, tmin: f32, tmax: f32) -> Self { Self { origin, direction: direction.normalize(), tmin, tmax } } 
    pub fn query(origin: Vec3, radius: f32) -> Self { Self { origin, direction: Vec3::new(1.0, 0.0, 0.0), tmin: 0.0, tmax: radius } }
}

#[derive(Debug, Clone, Copy)]
pub struct Hit { pub primitive_id: u32, pub distance: f32, pub position: Vec3 }
pub enum HitAction { Continue, Terminate, Ignore }

pub trait Geometry: Sized {
    fn intersect(&self, ray: &Ray) -> Option<Hit>;
    fn bounds(&self) -> AABB;
    fn name() -> &'static str;
    fn pack_optix(&self) -> Vec<f32>;
    fn get_hardware_triangle_buffers(_primitives: &[Self]) -> Option<(Vec<f32>, Vec<u32>)> { None }
    
    // NEW: Allow fast radius mutations for Loop Invariant Code Motion (LICM)
    fn set_radius(&mut self, _radius: f32) {} 
}

#[derive(Debug, Clone, Copy)]
pub struct Sphere { pub center: Vec3, pub radius: f32 }
impl Geometry for Sphere {
    fn name() -> &'static str { "Sphere" }
    fn pack_optix(&self) -> Vec<f32> { vec![self.center.x, self.center.y, self.center.z, self.radius] }
    fn bounds(&self) -> AABB { let r = Vec3::new(self.radius, self.radius, self.radius); AABB { min: self.center - r, max: self.center + r } }
    
    // NEW: Implement for Sphere
    fn set_radius(&mut self, radius: f32) { self.radius = radius; }
    
    fn intersect(&self, ray: &Ray) -> Option<Hit> {
        let dx = ray.origin.x - self.center.x;
        let dy = ray.origin.y - self.center.y;
        let dz = ray.origin.z - self.center.z;
        let dist_sq = dx*dx + dy*dy + dz*dz;
    
    	if dist_sq <= self.radius * self.radius {
            Some(Hit { primitive_id: 0, distance: dist_sq.sqrt(), position: ray.origin })
    	} else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Triangle { pub v0: Vec3, pub v1: Vec3, pub v2: Vec3 }
impl Geometry for Triangle {
    fn name() -> &'static str { "Triangle" }
    fn pack_optix(&self) -> Vec<f32> { vec![self.v0.x, self.v0.y, self.v0.z, self.v1.x, self.v1.y, self.v1.z, self.v2.x, self.v2.y, self.v2.z] }
    fn bounds(&self) -> AABB { let mut aabb = AABB::empty(); aabb.grow(self.v0); aabb.grow(self.v1); aabb.grow(self.v2); aabb }
    fn intersect(&self, ray: &Ray) -> Option<Hit> {
        let edge1 = self.v1 - self.v0; let edge2 = self.v2 - self.v0; let h = ray.direction.cross(&edge2);
        let a = edge1.dot(&h); if a > -1e-7 && a < 1e-7 { return None; }
        let f = 1.0 / a; let s = ray.origin - self.v0; let u = f * s.dot(&h);
        if u < 0.0 || u > 1.0 { return None; }
        let q = s.cross(&edge1); let v = f * ray.direction.dot(&q);
        if v < 0.0 || u + v > 1.0 { return None; }
        let t = f * edge2.dot(&q);
        if t > ray.tmin && t < ray.tmax { Some(Hit { primitive_id: 0, distance: t, position: ray.origin + ray.direction * t }) } else { None }
    }
    fn get_hardware_triangle_buffers(primitives: &[Self]) -> Option<(Vec<f32>, Vec<u32>)> {
        let mut verts = Vec::with_capacity(primitives.len() * 9);
        let mut indices = Vec::with_capacity(primitives.len() * 3);
        for (i, tri) in primitives.iter().enumerate() {
            verts.extend_from_slice(&[tri.v0.x, tri.v0.y, tri.v0.z, tri.v1.x, tri.v1.y, tri.v1.z, tri.v2.x, tri.v2.y, tri.v2.z]);
            indices.extend_from_slice(&[(i*3) as u32, (i*3+1) as u32, (i*3+2) as u32]);
        }
        Some((verts, indices))
    }
}

#[derive(Debug)]
pub struct BvhNode { pub bounds: AABB, pub left_first: u32, pub count: u32 }

pub struct Scene<G: Geometry> {
    pub primitives: Vec<G>,
    pub primitive_indices: Vec<usize>,
    pub nodes: Vec<BvhNode>,
    #[doc(hidden)] pub __gprt_id: u64,
    #[doc(hidden)] pub __gprt_is_dirty: bool,
}

impl<G: Geometry> Scene<G> {
    pub fn build<I>(primitives: I) -> Self where I: IntoIterator<Item = G> {
        let mut scene = Self { 
            primitives: primitives.into_iter().collect(), 
            primitive_indices: Vec::new(), 
            nodes: Vec::new(),
            __gprt_id: NEXT_SCENE_ID.fetch_add(1, Ordering::Relaxed),
            __gprt_is_dirty: true,
        };
        let prim_count = scene.primitives.len();
        if prim_count == 0 { return scene; }
        scene.primitive_indices = (0..prim_count).collect();
        scene.nodes.push(BvhNode { bounds: AABB::empty(), left_first: 0, count: prim_count as u32 });
        scene.update_node_bounds(0);
        scene.subdivide(0);
        scene
    }
    
    pub fn mark_dirty(&mut self) { self.__gprt_is_dirty = true; }

    fn update_node_bounds(&mut self, node_idx: usize) {
        let node = &mut self.nodes[node_idx];
        let mut bounds = AABB::empty();
        for i in 0..node.count {
            let prim_idx = self.primitive_indices[(node.left_first + i) as usize];
            bounds.merge(&self.primitives[prim_idx].bounds());
        }
        node.bounds = bounds;
    }

    fn subdivide(&mut self, node_idx: usize) {
        let (left_first, count) = {
            let node = &self.nodes[node_idx];
            if node.count <= 2 { return; }
            (node.left_first as usize, node.count as usize)
        };

        let bounds = self.nodes[node_idx].bounds;
        let extent = bounds.max - bounds.min;
        let mut axis = 0;
        if extent.y > extent.x { axis = 1; }
        if extent.z > extent.y && extent.z > extent.x { axis = 2; }

        let split_pos = bounds.center();
        let split_val = if axis == 0 { split_pos.x } else if axis == 1 { split_pos.y } else { split_pos.z };

        let mut i = left_first;
        let mut j = left_first + count - 1;
        while i <= j {
            let prim_idx = self.primitive_indices[i];
            let prim_center = self.primitives[prim_idx].bounds().center();
            let prim_val = if axis == 0 { prim_center.x } else if axis == 1 { prim_center.y } else { prim_center.z };
            
            if prim_val < split_val { i += 1; } 
            else {
                self.primitive_indices.swap(i, j);
                if j == 0 { break; }
                j -= 1;
            }
        }

        let left_count = i - left_first;
        if left_count == 0 || left_count == count { return; }

        let left_child_idx = self.nodes.len() as u32;
        self.nodes.push(BvhNode { bounds: AABB::empty(), left_first: left_first as u32, count: left_count as u32 });
        self.nodes.push(BvhNode { bounds: AABB::empty(), left_first: i as u32, count: (count - left_count) as u32 });

        self.nodes[node_idx].left_first = left_child_idx;
        self.nodes[node_idx].count = 0; 

        self.update_node_bounds(left_child_idx as usize);
        self.update_node_bounds((left_child_idx + 1) as usize);

        self.subdivide(left_child_idx as usize);
        self.subdivide((left_child_idx + 1) as usize);
    }

    pub fn traverse<F, C>(&self, ray: &Ray, mut any_hit_cb: F, mut closest_hit_cb: C)
    where F: FnMut(Hit) -> HitAction, C: FnMut(Hit) {
        if self.nodes.is_empty() { return; }
        let mut best_hit: Option<Hit> = None;
        let mut stack = [0usize; 64];
        let mut stack_ptr = 0;
        stack[stack_ptr] = 0; 
        let mut terminated = false;

        while stack_ptr < 64 && !terminated {
            let node_idx = stack[stack_ptr];
            if node_idx == usize::MAX { break; }
            stack[stack_ptr] = usize::MAX; 
            if stack_ptr > 0 { stack_ptr -= 1; }

            let node = &self.nodes[node_idx];
            if !node.bounds.intersect(ray) { continue; }

            if node.count > 0 { 
                for i in 0..node.count {
                    let prim_idx = self.primitive_indices[(node.left_first + i) as usize];
                    if let Some(mut hit) = self.primitives[prim_idx].intersect(ray) {
                        hit.primitive_id = prim_idx as u32;
                        match any_hit_cb(hit) {
                            HitAction::Terminate => { terminated = true; break; }
                            HitAction::Ignore => continue,
                            HitAction::Continue => {
                                match best_hit {
                                    None => best_hit = Some(hit),
                                    Some(current) if hit.distance < current.distance => best_hit = Some(hit),
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            } else { 
                stack_ptr += 1; stack[stack_ptr] = node.left_first as usize;
                stack_ptr += 1; stack[stack_ptr] = (node.left_first + 1) as usize;
            }
        }

        if !terminated {
            if let Some(hit) = best_hit { closest_hit_cb(hit); }
        }
    }
}



fn expand_bits(v: u32) -> u32 {
    let mut v = v & 0x000003ff; // 10 bits
    v = (v | (v << 16)) & 0x030000ff;
    v = (v | (v <<  8)) & 0x0300f00f;
    v = (v | (v <<  4)) & 0x030c30c3;
    v = (v | (v <<  2)) & 0x09249249;
    v
}

fn morton3d(x: u32, y: u32, z: u32) -> u32 {
    (expand_bits(x) << 2) | (expand_bits(y) << 1) | expand_bits(z)
}

const BUCKET_SIZE: usize = 128;
const MIN_SPLIT_SIZE: f32 = 1e-5f32;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NodeSpatial {
    pub com: Vec3,
    pub mass: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NodeRouting {
    pub next_idx: u32,
    pub autorope: u32,
    pub is_leaf: u32,
    pub width: f32,
    pub bucket_start: u32,
    pub num_particles: u32,
    pub _pad: [u32; 2],     
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FlatNode {
    pub com: Vec3,
    pub mass: f32,
    pub width: f32,
    pub is_leaf: u32,
    pub next_idx: u32,
    pub autorope: u32,
    pub bucket_start: u32,
    pub num_particles: u32,
}


#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Body {
    pub pos: Vec3,
    pub mass: f32,
    pub velocity: Vec3,
}

struct OctreeNode {
    bbox: AABB,
    children: Vec<usize>,
    com: Vec3,
    mass: f32,
    body_start: usize,
    body_count: usize,
}

pub struct BarnesHutTree {
    pub nodes: Vec<FlatNode>,
    pub bodies: Vec<Body>, 
}


struct TreeBuilder<'a> {
    bodies: &'a [Body],
    arena: Vec<OctreeNode>,
    scratch: Vec<usize>, 
}

impl<'a> TreeBuilder<'a> {
    fn build_recursive(
        &mut self,
        indices: &mut [usize], 
        bbox: AABB,
        current_width: f32,
        abs_start_offset: usize,
    ) -> usize {
        let node_idx = self.arena.len();
        let count = indices.len();
        let is_leaf = count <= BUCKET_SIZE || current_width < MIN_SPLIT_SIZE;
        
        self.arena.push(OctreeNode {
            bbox,
            children: Vec::new(),
            com: Vec3::new(0.0, 0.0, 0.0),
            mass: 0.0,
            body_start: abs_start_offset,
            body_count: if is_leaf { count } else { 0 }, 
        });

        if is_leaf { return node_idx; }

        let center = bbox.center();
        
        let mut counts = [0usize; 8];
        for &i in indices.iter() {
            let b = &self.bodies[i];
            let mut oct = 0;
            if b.pos.x >= center.x { oct |= 1; }
            if b.pos.y >= center.y { oct |= 2; }
            if b.pos.z >= center.z { oct |= 4; }
            counts[oct] += 1;
        }
        
        let mut offsets = [0usize; 8];
        for i in 1..8 { offsets[i] = offsets[i-1] + counts[i-1]; }
        
        let scratch_slice = &mut self.scratch[abs_start_offset .. abs_start_offset + count];
        let mut current_offsets = offsets;
        for &i in indices.iter() {
            let b = &self.bodies[i];
            let mut oct = 0;
            if b.pos.x >= center.x { oct |= 1; }
            if b.pos.y >= center.y { oct |= 2; }
            if b.pos.z >= center.z { oct |= 4; }
            scratch_slice[current_offsets[oct]] = i;
            current_offsets[oct] += 1;
        }
        
        indices.copy_from_slice(scratch_slice);
        
        let mut children = Vec::new();
        let child_width = current_width * 0.5;
        
        let mut current_rel_offset = 0; 
        let mut current_abs_offset = abs_start_offset; 
        
        for oct in 0..8 {
            let c_count = counts[oct];
            if c_count > 0 {
                let min_x = if (oct & 1) == 0 { bbox.min.x } else { center.x };
                let max_x = if (oct & 1) == 0 { center.x } else { bbox.max.x };
                let min_y = if (oct & 2) == 0 { bbox.min.y } else { center.y };
                let max_y = if (oct & 2) == 0 { center.y } else { bbox.max.y };
                let min_z = if (oct & 4) == 0 { bbox.min.z } else { center.z };
                let max_z = if (oct & 4) == 0 { center.z } else { bbox.max.z };
                let child_bbox = AABB { min: Vec3::new(min_x, min_y, min_z), max: Vec3::new(max_x, max_y, max_z) };
                
                let child_idx = self.build_recursive(
                    &mut indices[current_rel_offset .. current_rel_offset + c_count], 
                    child_bbox,
                    child_width,
                    current_abs_offset, 
                );
                children.push(child_idx);
                
                current_rel_offset += c_count;
                current_abs_offset += c_count;
            }
        }
        
        self.arena[node_idx].children = children;
        node_idx
    }
}

impl BarnesHutTree {
    pub fn build(bodies: &[Body]) -> Self {
        if bodies.is_empty() { return Self { nodes: Vec::new(), bodies: Vec::new() }; }

        let mut min_b = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max_b = Vec3::new(f32::MIN, f32::MIN, f32::MIN);
        for b in bodies { min_b = min_b.min(&b.pos); max_b = max_b.max(&b.pos); }
        
        let extent = max_b - min_b;
        let max_extent = extent.x.max(extent.y).max(extent.z);
        let center = Vec3::new((min_b.x + max_b.x) * 0.5, (min_b.y + max_b.y) * 0.5, (min_b.z + max_b.z) * 0.5);
        
        let mut root_bbox = AABB::empty();
        root_bbox.min = Vec3::new(center.x - max_extent/2.0, center.y - max_extent/2.0, center.z - max_extent/2.0);
        root_bbox.max = Vec3::new(center.x + max_extent/2.0, center.y + max_extent/2.0, center.z + max_extent/2.0);

        let scale = if max_extent > 0.0 { 1023.0 / max_extent } else { 0.0 };

        let mut indices: Vec<usize> = (0..bodies.len()).collect();
        indices.sort_unstable_by_key(|&i| {
            let b = &bodies[i];
            let lx = ((b.pos.x - min_b.x) * scale) as u32;
            let ly = ((b.pos.y - min_b.y) * scale) as u32;
            let lz = ((b.pos.z - min_b.z) * scale) as u32;
            morton3d(lx, ly, lz)
        });

        let mut builder = TreeBuilder {
            bodies,
            arena: Vec::new(),
            scratch: vec![0; bodies.len()], 
        };
        
        builder.build_recursive(&mut indices, root_bbox, max_extent, 0);

        if !builder.arena.is_empty() { Self::compute_com(0, &mut builder.arena, bodies, &indices); }

        let mut flat_nodes = Vec::new();
        let mut flat_bodies = Vec::new();
        if !builder.arena.is_empty() { Self::flatten_dfs(0, &builder.arena, &mut flat_nodes, &mut flat_bodies, bodies, &indices); }

        Self { nodes: flat_nodes, bodies: flat_bodies }
    }

    fn compute_com(node_idx: usize, arena: &mut Vec<OctreeNode>, bodies: &[Body], indices: &[usize]) {
        if arena[node_idx].children.is_empty() {
            let mut total_mass = 0.0;
            let mut com = Vec3::new(0.0, 0.0, 0.0);
            let start = arena[node_idx].body_start;
            let count = arena[node_idx].body_count;
            for i in 0..count {
                let b = &bodies[indices[start + i]];
                total_mass += b.mass;
                com = com + (b.pos * b.mass);
            }
            arena[node_idx].mass = total_mass;
            if total_mass > 0.0 { arena[node_idx].com = com / total_mass; }
            return;
        }

        let mut total_mass = 0.0;
        let mut com = Vec3::new(0.0, 0.0, 0.0);
        let children = arena[node_idx].children.clone();

        for child_idx in children {
            Self::compute_com(child_idx, arena, bodies, indices);
            let child_mass = arena[child_idx].mass;
            let child_com = arena[child_idx].com;
            total_mass += child_mass;
            com = com + (child_com * child_mass);
        }

        arena[node_idx].mass = total_mass;
        if total_mass > 0.0 { arena[node_idx].com = com / total_mass; }
    }

    fn flatten_dfs(node_idx: usize, arena: &[OctreeNode], flat_array: &mut Vec<FlatNode>, flat_bodies: &mut Vec<Body>, bodies: &[Body], indices: &[usize]) -> usize {
        let current_flat_idx = flat_array.len();
        let node = &arena[node_idx];
        let extent = node.bbox.max - node.bbox.min;
        let width = extent.x.max(extent.y).max(extent.z);

        let mut bucket_start = 0;
        let num_particles = node.body_count as u32;
        
        if num_particles > 0 {
            bucket_start = flat_bodies.len() as u32;
            let start = node.body_start;
            for i in 0..node.body_count {
                flat_bodies.push(bodies[indices[start + i]]);
            }
        }

        flat_array.push(FlatNode {
            com: node.com,
            mass: node.mass,
            width,
            is_leaf: if node.children.is_empty() { 1 } else { 0 },
            next_idx: (current_flat_idx + 1) as u32,
            autorope: 0, 
            bucket_start,
            num_particles,
        });

        for &child_idx in &node.children {
            Self::flatten_dfs(child_idx, arena, flat_array, flat_bodies, bodies, indices);
        }

        let autorope_idx = flat_array.len() as u32;
        flat_array[current_flat_idx].autorope = autorope_idx;

        flat_array.len() - current_flat_idx
    }
}
