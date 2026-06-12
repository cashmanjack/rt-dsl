use std::ops::{Add, Sub, Mul, Div};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_SCENE_ID: AtomicU64 = AtomicU64::new(1);

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
}

#[derive(Debug, Clone, Copy)]
pub struct Sphere { pub center: Vec3, pub radius: f32 }
impl Geometry for Sphere {
    fn name() -> &'static str { "Sphere" }
    fn pack_optix(&self) -> Vec<f32> { vec![self.center.x, self.center.y, self.center.z, self.radius] }
    fn bounds(&self) -> AABB { let r = Vec3::new(self.radius, self.radius, self.radius); AABB { min: self.center - r, max: self.center + r } }
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
