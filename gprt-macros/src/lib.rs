use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ExprClosure, Ident, Token, parse::{Parse, ParseStream}, Expr, visit::Visit};
use std::collections::{HashMap, HashSet};

// ==========================================
// 1. Ray Traversal Core
// ==========================================
struct RayTraversalInput {
    rays: Ident, 
    scene: Ident,
    any_hit: Option<ExprClosure>, 
    closest_hit: Option<ExprClosure>,
}

impl Parse for RayTraversalInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let rays: Ident = input.parse()?; input.parse::<Token![,]>()?;
        let scene: Ident = input.parse()?; 
        
        let mut any_hit = None; 
        let mut closest_hit = None;
        
        while input.peek(Token![,]) {
            input.parse::<Token![,]>()?; 
            if input.is_empty() { break; }
            
            let ident: Ident = input.parse()?; 
            input.parse::<Token![=]>()?;
            let closure: ExprClosure = input.parse()?;
            
            if ident == "any_hit" { any_hit = Some(closure); } 
            else if ident == "closest_hit" { closest_hit = Some(closure); }
        }
        
        Ok(RayTraversalInput { rays, scene, any_hit, closest_hit })
    }
}

struct PayloadVisitor { pub arrays: HashSet<String>, pub values: HashSet<String> }
impl<'ast> Visit<'ast> for PayloadVisitor {
    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        if node.method == "push" {
            if let Expr::Path(p) = &*node.receiver {
                if let Some(ident) = p.path.get_ident() { self.arrays.insert(ident.to_string()); }
            }
        }
        syn::visit::visit_expr_method_call(self, node);
    }
    fn visit_expr_assign(&mut self, node: &'ast syn::ExprAssign) {
        if let Expr::Path(p) = &*node.left {
            if let Some(ident) = p.path.get_ident() { self.values.insert(ident.to_string()); }
        }
        syn::visit::visit_expr_assign(self, node);
    }
}

#[proc_macro]
pub fn ray_traversal(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as RayTraversalInput);
    let rays_ident = &input.rays; 
    let scene_ident = &input.scene;

    let mut visitor = PayloadVisitor { arrays: HashSet::new(), values: HashSet::new() };
    if let Some(c) = &input.any_hit { visitor.visit_expr_closure(c); }
    if let Some(c) = &input.closest_hit { visitor.visit_expr_closure(c); }

    let mut array_indices = HashMap::new();
    for (i, name) in visitor.arrays.iter().enumerate() { array_indices.insert(name.clone(), i); }
    let mut value_indices = HashMap::new();
    for (i, name) in visitor.values.iter().enumerate() { value_indices.insert(name.clone(), i); }

    let geom_type = "Sphere".to_string(); 

    let temp_ir = gprt_ir::RtProgram {
        array_indices, value_indices,
        any_hit_tokens: input.any_hit.as_ref().map(|c| { let body = &c.body; quote!(#body) }),
        closest_hit_tokens: input.closest_hit.as_ref().map(|c| { let body = &c.body; quote!(#body) }),
        geom_type: geom_type.clone(),
    };

    // FIX: Scope cuda_source to avoid unused variable warnings
    #[cfg(feature = "optix")]
    let ir_bytes = {
        let cuda_source = gprt_codegen::generate_cuda_source(&temp_ir);
        gprt_codegen::compile_to_optixir(&cuda_source)
    };
    
    #[cfg(not(feature = "optix"))]
    let ir_bytes: Vec<u8> = Vec::new();

    let is_hw_triangle = geom_type == "Triangle";

    let array_names: Vec<String> = visitor.arrays.into_iter().collect();
    let value_names: Vec<String> = visitor.values.into_iter().collect();
    
    let array_idents: Vec<Ident> = array_names.iter().map(|n| syn::parse_str(n).unwrap()).collect();
    let value_idents: Vec<Ident> = value_names.iter().map(|n| syn::parse_str(n).unwrap()).collect();
    
    let temp_value_idents: Vec<Ident> = value_names.iter().map(|n| {
        syn::parse_str(&format!("__gprt_temp_{}", n)).unwrap()
    }).collect();

    let mut type_checks = quote! {};
    let cpu_any_hit = if let Some(ref ah) = input.any_hit { quote! { #ah } } else { quote! { |_hit| gprt_core::HitAction::Continue } };
    let cpu_closest_hit = if let Some(ref ch) = input.closest_hit { quote! { #ch } } else { quote! { |_hit| {} } };

    if let Some(ref ah) = input.any_hit {
        type_checks = quote! { #type_checks let _: &mut dyn FnMut(gprt_core::Hit) -> gprt_core::HitAction = &mut #ah; };
    }
    if let Some(ref ch) = input.closest_hit {
        type_checks = quote! { #type_checks let _: &mut dyn FnMut(gprt_core::Hit) = &mut #ch; };
    }

    let expanded = quote! {
        {
            const IR_BYTES: &[u8] = &[#(#ir_bytes),*];
            const IS_HW_TRIANGLE: bool = #is_hw_triangle;
            
            #[cfg(feature = "optix")]
            {
                #type_checks
                static __GPRT_PIPELINE_CACHE: std::sync::OnceLock<gprt_optix::OptixPipeline> = std::sync::OnceLock::new();
                let pipeline = __GPRT_PIPELINE_CACHE.get_or_init(|| {
                    gprt_optix::OptixPipeline::new(IR_BYTES, IS_HW_TRIANGLE)
                });
                
                #( pipeline.register_array(#array_names, &mut #array_idents); )*
                #( let mut #temp_value_idents: u32 = 0; )*
                #( pipeline.register_value(#value_names, &mut #temp_value_idents); )*
                
                pipeline.trace_scene(&mut #scene_ident, &#rays_ident);
                
                #( pipeline.retrieve_array(#array_names, &mut #array_idents); )*
                #( 
                    let __gprt_res = pipeline.retrieve_value(#value_names);
                    if __gprt_res != u32::MAX { #value_idents = Some(__gprt_res); } 
                    else { #value_idents = None; }
                )*
            }

            #[cfg(not(feature = "optix"))]
            {
                #type_checks
                for ray in #rays_ident.iter() {
                    #scene_ident.traverse(ray, #cpu_any_hit, #cpu_closest_hit);
                }
            }
        }
    };
    TokenStream::from(expanded)
}

// ==========================================
// 2. High-Level Spatial Verbs
// ==========================================
struct NnInput {
    data: Expr,
    queries: Expr,
    param: Expr,
    output: Expr,
}

impl Parse for NnInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let data: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let queries: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let param: Expr = input.parse()?; input.parse::<Token![,]>()?;
        let output: Expr = input.parse()?;
        Ok(NnInput { data, queries, param, output })
    }
}

#[proc_macro]
pub fn r_nn(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as NnInput);
    let data = input.data;
    let queries = input.queries;
    let radius = input.param;
    let output = input.output;

    let expanded = quote! {
        {
            let mut __gprt_scene = gprt_core::Scene::build(
                #data.iter().map(|p| gprt_core::Sphere { center: *p, radius: #radius })
            );
            let __gprt_rays: Vec<gprt_core::Ray> = #queries.iter()
                .map(|q| gprt_core::Ray::query(*q, #radius))
                .collect();
            
            ray_traversal!(
                __gprt_rays,
                __gprt_scene,
                any_hit = |hit| {
                    #output.push(hit.primitive_id);
                    gprt_core::HitAction::Continue
                }
            );
        }
    };
    TokenStream::from(expanded)
}





#[proc_macro]
pub fn k_nn(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as NnInput);
    let data = input.data;
    let queries = input.queries;
    let k = input.param;
    let output = input.output;

    let any_hit_body: proc_macro2::TokenStream = quote! {
        { knn_out.push(hit.primitive_id); gprt_core::HitAction::Continue }
    };
    
    let temp_ir = gprt_ir::RtProgram {
        array_indices: [("knn_out".to_string(), 0)].into_iter().collect(),
        value_indices: std::collections::HashMap::new(),
        any_hit_tokens: Some(any_hit_body),
        closest_hit_tokens: None,
        geom_type: "Sphere".to_string(),
    };
    
    #[cfg(feature = "optix")]
    let ir_bytes = {
        let cuda_source = gprt_codegen::generate_cuda_source(&temp_ir);
        gprt_codegen::compile_to_optixir(&cuda_source)
    };
    
    #[cfg(not(feature = "optix"))]
    let ir_bytes: Vec<u8> = Vec::new();

    let expanded = quote! {
        {
            #[cfg(feature = "optix")]
            {
                const __GPRT_KNN_IR: &[u8] = &[#(#ir_bytes),*];
                let __n_data = #data.len();
                let __n_queries = #queries.len();

                // 1. LCG Radius Sampler (95th Percentile Heuristic)
                let mut __seed: u64 = 0x123456789ABCDEF;
                let mut __next_rand = || -> usize {
                    __seed = __seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (__seed >> 33) as usize
                };
                let __sample_size = 5000.min(__n_data);
                let __query_sample_size = 100.min(__n_queries);
                
                let mut __sampled_dists: Vec<f32> = Vec::with_capacity(__query_sample_size);
                for _ in 0..__query_sample_size {
                    let __qi = __next_rand() % __n_queries;
                    let __q = #queries[__qi];
                    let mut __dists = [f32::MAX; 4]; 
                    for _ in 0..__sample_size {
                        let __pi = __next_rand() % __n_data;
                        let __p = #data[__pi];
                        let __dx = __q.x - __p.x; let __dy = __q.y - __p.y; let __dz = __q.z - __p.z;
                        let __d = (__dx*__dx + __dy*__dy + __dz*__dz).sqrt();
                        if __d < __dists[3] {
                            __dists[3] = __d;
                            if __dists[3] < __dists[2] { __dists.swap(2, 3); }
                            if __dists[2] < __dists[1] { __dists.swap(1, 2); }
                            if __dists[1] < __dists[0] { __dists.swap(0, 1); }
                        }
                    }
                    if __dists[3] != f32::MAX { __sampled_dists.push(__dists[3]); }
                }
                
                __sampled_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let __p95_idx = ((__sampled_dists.len() as f32) * 0.95) as usize;
                let __chosen_dist = if __sampled_dists.is_empty() { 10.0 } else { __sampled_dists[__p95_idx.min(__sampled_dists.len() - 1)] };
                
                // Start at 1/8th (2^-3) of the 95th percentile radius
                let mut __current_radius = __chosen_dist * 1.1 * 0.0625; 

                static __GPRT_KNN_PIPELINE: std::sync::OnceLock<gprt_optix::OptixPipeline> = std::sync::OnceLock::new();
                let __pipeline = __GPRT_KNN_PIPELINE.get_or_init(|| {
                    gprt_optix::OptixPipeline::new(__GPRT_KNN_IR, false) 
                });
                
                let mut __scene = gprt_core::Scene::build(
                    #data.iter().map(|__p| gprt_core::Sphere { center: *__p, radius: __current_radius })
                );
                
                let mut __active_indices: Vec<usize> = (0..__n_queries).collect();
                let mut __per_query_results: Vec<Vec<u32>> = vec![Vec::new(); __n_queries];
                
                let mut __iteration = 0;
                loop {
                    // EXACT / UNBOUNDED MODE: No iteration cap. Runs until 100% recall is achieved.
                    if __active_indices.is_empty() { break; } 
                    
                    // Safety valve to prevent infinite loops on degenerate hardware states
                    if __iteration > 30 { 
                        eprintln!("[TrueKNN] Safety exit at 30 rounds: {} extreme outliers skipped.", __active_indices.len());
                        break; 
                    }

                    __iteration += 1;
                    let __n_active = __active_indices.len();
                    
                    eprintln!("[TrueKNN Exact] Round {}: {} queries remaining, radius = {:.4}", __iteration, __n_active, __current_radius);

                    let __dynamic_cap: usize = if __n_active > 100_000 { 1_000 } 
                        else if __n_active > 10_000 { 5_000 } 
                        else if __n_active > 1_000 { 20_000 } 
                        else { 50_000 };

                    __pipeline.register_array_batched("knn_out", __dynamic_cap as u32, __n_active);
                    
                    let __active_rays: Vec<gprt_core::Ray> = __active_indices.iter()
                        .map(|&qi| gprt_core::Ray::query(#queries[qi], __current_radius))
                        .collect();
                    
                    __pipeline.trace_scene(&mut __scene, &__active_rays);
                    
                    let __batched_results = __pipeline.retrieve_array_batched("knn_out", __n_active, __dynamic_cap);
                    
                    let mut __next_active = Vec::new();
                    for (local_idx, &global_qi) in __active_indices.iter().enumerate() {
                        let __q = #queries[global_qi];
                        let mut __candidates: Vec<(u32, f32)> = __batched_results[local_idx].iter()
                            .map(|&prim_id| (prim_id, #data[prim_id as usize].distance(&__q)))
                            .collect();
                        __candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        
                        if __candidates.len() >= #k {
                            __candidates.truncate(#k);
                            for (id, _) in __candidates { __per_query_results[global_qi].push(id); }
                        } else if __batched_results[local_idx].len() >= __dynamic_cap - 100 {
                            for (id, _) in __candidates { __per_query_results[global_qi].push(id); }
                        } else {
                            __next_active.push(global_qi);
                        }
                    }
                    __active_indices = __next_active;
                    if __active_indices.is_empty() { break; }
                    
                    __current_radius *= 2.0;
                    for __prim in &mut __scene.primitives { __prim.radius = __current_radius; }
                    __scene.mark_dirty();
                }
                
                #output.clear();
                for __res in __per_query_results {
                    for __id in __res { #output.push(__id); }
                }
            }

            #[cfg(not(feature = "optix"))]
            { panic!("k_nn! requires the 'optix' feature to be enabled."); }
        }
    };
    TokenStream::from(expanded)
}
