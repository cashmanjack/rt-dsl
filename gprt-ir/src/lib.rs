use proc_macro2::TokenStream;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ==========================================
// 1. THE SHADER AST (Device-Side Logic)
// ==========================================
#[derive(Debug, Clone)]
pub enum ShaderNode {
    TraceRay { tmax: TokenStream, on_hit: Box<ShaderNode>, on_miss: Box<ShaderNode> },
    TerminateRay,
    IgnoreIntersection,
    PushToDynamicArray { array_name: String, value: TokenStream },
    UpdatePayload { register: usize, value: TokenStream },
    If { condition: TokenStream, then_body: Box<ShaderNode>, else_body: Box<ShaderNode> },
    Block(Vec<ShaderNode>),
    RawCuda(String), 
}

// ==========================================
// 2. THE HOST STRATEGY (CPU-Side Mapping)
// ==========================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HostStrategy {
    StandardBVH { geom_type: String },
    TriangleHack { autorope_tree: bool },
}

// ==========================================
// 3. THE SCHEDULE (Tunable Knobs for the Profiler)
// ==========================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RadiusHeuristic {
    SampledMax,           
    SampledPercentile(f32),
    Fixed(f32),          
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemoryStrategy {
    GlobalMemory,        
    PayloadRegisterHeap, 
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BvhBuildStrategy {
    PreferFastTrace, // SAH Optimization (Slow Build, Fast Trace)
    PreferFastBuild, // Linear Split (Fast Build, Slow Trace)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GeometryType {
    Spheres,
    Triangles,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schedule {
    pub radius_increment_mult: f32,
    pub max_hits_per_query: u32,
    pub use_morton_lbv: bool,
    pub radius_heuristic: RadiusHeuristic,
    pub memory_strategy: MemoryStrategy,
    pub build_strategy: BvhBuildStrategy, // EXPOSED BUILD STRATEGY
    pub geom_type: GeometryType,          // EXPOSED GEOMETRY TYPE
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntersectionImpl { AnyHitProgram, IntersectionProgram }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BvhUpdateMode { Refit, Rebuild }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BufferCapacity { Dynamic, Static(usize), PowerOfTwo }

impl Default for Schedule {
    fn default() -> Self {
        Self {
            radius_heuristic: RadiusHeuristic::SampledPercentile(0.10),
            radius_increment_mult: 3.0,
            max_hits_per_query: 2000,
            use_morton_lbv: false,
            memory_strategy: MemoryStrategy::PayloadRegisterHeap,
            build_strategy: BvhBuildStrategy::PreferFastTrace, // Default to high quality SAH
            geom_type: GeometryType::Spheres,                  // Default to Spheres
        }
    }
}

// ==========================================
// 4. THE UNIFIED PROGRAM IR
// ==========================================
#[derive(Debug, Clone)]
pub struct RtProgram {
    pub raygen_body: ShaderNode,
    pub miss_body: Option<ShaderNode>,
    pub anyhit_body: Option<ShaderNode>,
    pub closesthit_body: Option<ShaderNode>,
    pub intersection_body: Option<ShaderNode>,

    pub payload_layout: Vec<String>,
    pub schedule: Schedule,
    pub array_indices: HashMap<String, usize>,
}
