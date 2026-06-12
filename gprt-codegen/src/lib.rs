use gprt_ir::RtProgram;
use syn::{Expr, ExprMethodCall, ExprField, ExprPath, ExprAssign, ExprCall, Member, BinOp, ExprBinary, ExprLit, ExprParen, ExprBlock, Stmt};
use quote::quote;
use std::collections::HashMap;

pub struct CodegenContext<'a> { pub array_indices: &'a HashMap<String, usize>, pub value_indices: &'a HashMap<String, usize> }

pub fn translate_expr(ctx: &CodegenContext, expr: &Expr) -> String {
    match expr {
        Expr::Block(ExprBlock { block, .. }) => {
            let mut stmts = Vec::new();
            for stmt in &block.stmts {
                match stmt {
                    Stmt::Expr(e, _) => stmts.push(format!("{};", translate_expr(ctx, e))),
                    _ => {}
                }
            }
            format!("{{ {} }}", stmts.join(" "))
        }
        Expr::MethodCall(ExprMethodCall { receiver, method, args, .. }) => {
            if method == "push" {
                let vec_name = translate_expr(ctx, receiver);
                if let Some(&idx) = ctx.array_indices.get(&vec_name) {
                    let val = translate_expr(ctx, &args[0]);
                    // 2D Demultiplexing: Each ray writes to its own slice using query_id

return format!(
    "{{ 
        unsigned int __qid = optixGetLaunchIndex().x; 
        unsigned int __idx = atomicAdd((unsigned int*)bundle->dyn_lens[{}] + __qid, 1u); 
        if (__idx < bundle->dyn_caps[{}]) {{ 
            ((unsigned int*)bundle->dyn_ptrs[{}])[__qid * bundle->dyn_caps[{}] + __idx] = {}; 
        }} 
    }}",
    idx, idx, idx, idx, val
);

                }
            }
            format!("{}.{}({})", translate_expr(ctx, receiver), method, args.iter().map(|a| translate_expr(ctx, a)).collect::<Vec<_>>().join(", "))
        }
        Expr::Field(ExprField { base, member, .. }) => {
            let base_str = translate_expr(ctx, base);
            if let Member::Named(ident) = member {
                if base_str == "hit" && ident == "primitive_id" { return "hit_primitive_id".to_string(); }
                return format!("{}.{}", base_str, ident);
            }
            quote!(#expr).to_string().replace(" ", "")
        }
        Expr::Path(ExprPath { path, .. }) => {
            let path_str = quote!(#path).to_string().replace(" ", "");
            if path_str == "HitAction::Continue" || path_str == "gprt_core::HitAction::Continue" {
                return "optixIgnoreIntersection()".to_string();
            }
            if path_str == "HitAction::Terminate" || path_str == "gprt_core::HitAction::Terminate" {
                return "optixTerminateRay()".to_string();
            }
            if path_str == "HitAction::Ignore" || path_str == "gprt_core::HitAction::Ignore" {
                return "optixIgnoreIntersection()".to_string();
            }
            path_str
        }
        Expr::Assign(ExprAssign { left, right, .. }) => {
            let left_str = translate_expr(ctx, left);
            if let Some(&idx) = ctx.value_indices.get(&left_str) {
                let right_str = translate_expr(ctx, right);
                return format!("*(unsigned int*)bundle->val_ptrs[{}] = {}", idx, right_str);
            }
            format!("{} = {}", left_str, translate_expr(ctx, right))
        }
        Expr::Call(ExprCall { func, args, .. }) => {
            let func_str = quote!(#func).to_string().replace(" ", "");
            if func_str == "Some" && !args.is_empty() { return translate_expr(ctx, &args[0]); }
            format!("{}({})", translate_expr(ctx, func), args.iter().map(|a| translate_expr(ctx, a)).collect::<Vec<_>>().join(", "))
        }
        Expr::Binary(ExprBinary { left, op, right, .. }) => {
            let op_str = match op {
                BinOp::Add(_) => "+", BinOp::Sub(_) => "-", BinOp::Mul(_) => "*", BinOp::Div(_) => "/",
                _ => "?",
            };
            format!("({} {} {})", translate_expr(ctx, left), op_str, translate_expr(ctx, right))
        }
        Expr::Lit(ExprLit { lit, .. }) => quote!(#lit).to_string(),
        Expr::Paren(ExprParen { expr, .. }) => format!("({})", translate_expr(ctx, expr)),
        _ => quote!(#expr).to_string().replace(" ", ""),
    }
}




pub fn generate_cuda_source(ir: &RtProgram) -> String {
    let mut code = String::new();
    code.push_str("#include <optix.h>\n#include <optix_device.h>\n#include <vector_types.h>\n#include <vector_functions.h>\n\n");
    code.push_str("inline __device__ float gprt_dot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }\n\n");

    code.push_str("struct PayloadBundle { unsigned long long dyn_ptrs[8]; unsigned int dyn_caps[8]; unsigned long long dyn_lens[8]; unsigned long long val_ptrs[8]; };\n");
    code.push_str("struct LaunchParams { OptixTraversableHandle handle; float4* rays; int num_rays; PayloadBundle* bundle; };\n");
    code.push_str("__constant__ LaunchParams params;\n\n");

    let has_closest_hit = ir.closest_hit_tokens.is_some();
    let has_any_hit_continue = ir.any_hit_tokens.as_ref().map_or(false, |ts| ts.to_string().contains("HitAction::Continue"));
    let use_payload_tracking = has_closest_hit && has_any_hit_continue;
    

    code.push_str("extern \"C\" __global__ void __raygen__rg() {\n");
    code.push_str("\tuint3 launch_idx = optixGetLaunchIndex();\n\tint idx = launch_idx.x;\n\tif (idx >= params.num_rays) return;\n");
    code.push_str("\tfloat4 r = params.rays[idx];\n\tfloat3 origin = make_float3(r.x, r.y, r.z);\n\tfloat3 direction = make_float3(1.0f, 0.0f, 0.0f);\n\tfloat tmax = r.w;\n");
    
    if use_payload_tracking {
        code.push_str("\tunsigned int p0 = __float_as_uint(tmax);\n");
        code.push_str("\tunsigned int p1 = 0xFFFFFFFF;\n");
        code.push_str("\toptixTrace(params.handle, origin, direction, 0.0f, tmax, 0.0f, 1u, OPTIX_RAY_FLAG_NONE, 0u, 1u, 0u, p0, p1);\n");
        code.push_str("\tif (p1 != 0xFFFFFFFF) {\n");
        code.push_str("\t\tunsigned int hit_primitive_id = p1;\n");
        code.push_str("\t\tPayloadBundle* bundle = params.bundle; (void)bundle;\n");
        let closest_hit_expr = ir.closest_hit_tokens.as_ref().map(|ts| syn::parse2::<Expr>(ts.clone()).unwrap());
        let ctx = CodegenContext { array_indices: &ir.array_indices, value_indices: &ir.value_indices };
        if let Some(expr) = &closest_hit_expr {
            code.push_str(&format!("\t\t{}\n", translate_expr(&ctx, expr)));
        }
        code.push_str("\t}\n");
    } else {
        code.push_str("\toptixTrace(params.handle, origin, direction, 0.0f, tmax, 0.0f, 1u, OPTIX_RAY_FLAG_NONE, 0u, 1u, 0u);\n");
    }
    code.push_str("}\n\n");
    
    code.push_str("extern \"C\" __global__ void __miss__ms() {}\n\n");

    if ir.geom_type == "Sphere" {
        code.push_str("struct SphereSbtData { float4* spheres; char padding[8]; };\n");
        code.push_str("extern \"C\" __global__ void __intersection__is() {\n");
        code.push_str("\tconst SphereSbtData* sbt = (const SphereSbtData*)optixGetSbtDataPointer();\n");
        code.push_str("\tunsigned int prim_id = optixGetPrimitiveIndex();\n");
        code.push_str("\tfloat4 sphere = sbt->spheres[prim_id];\n");

        code.push_str("\tfloat3 o = optixGetObjectRayOrigin();\n"); // The query point
        code.push_str("\tfloat3 center = make_float3(sphere.x, sphere.y, sphere.z);\n");
        code.push_str("\tfloat dx = o.x - center.x;\n");
        code.push_str("\tfloat dy = o.y - center.y;\n");
        code.push_str("\tfloat dz = o.z - center.z;\n");
        code.push_str("\tfloat dist_sq = dx*dx + dy*dy + dz*dz;\n");
        code.push_str("\tfloat max_dist = optixGetRayTmax();\n"); // Read radius from Ray!
        code.push_str("\tif (dist_sq <= max_dist * max_dist) {\n");
        code.push_str("\t\toptixReportIntersection(max_dist, 0);\n"); // Dummy 't' value to trigger AnyHit
        code.push_str("\t}\n}\n\n");
    } else if ir.geom_type == "Triangle" {
        code.push_str("struct TriangleSbtData { float3* vertices; unsigned int* indices; };\n\n");
        // NO __intersection__is generated! Hardware handles it.
    }

    let any_hit_expr = ir.any_hit_tokens.as_ref().map(|ts| syn::parse2::<Expr>(ts.clone()).unwrap());
    let ctx = CodegenContext { array_indices: &ir.array_indices, value_indices: &ir.value_indices };

    code.push_str("extern \"C\" __global__ void __anyhit__ah() {\n");
    code.push_str("\tunsigned int hit_primitive_id = optixGetPrimitiveIndex();\n");
    code.push_str("\tPayloadBundle* bundle = params.bundle; (void)sizeof(bundle);\n");
    code.push_str("\t(void)sizeof(bundle); // Suppress NVCC #550\n\n");

    if use_payload_tracking {
        code.push_str("\tfloat current_min_t = __uint_as_float(optixGetPayload_0());\n");
        code.push_str("\tfloat t = optixGetRayTmax();\n");
        code.push_str("\tif (t < current_min_t) {\n");
        code.push_str("\t\toptixSetPayload_0(__float_as_uint(t));\n");
        code.push_str("\t\toptixSetPayload_1(hit_primitive_id);\n");
        code.push_str("\t}\n");
    }

    if let Some(expr) = &any_hit_expr {
        code.push_str(&format!("\t{}\n", translate_expr(&ctx, expr)));
    }
    code.push_str("}\n\n");

    if use_payload_tracking {
        code.push_str("extern \"C\" __global__ void __closesthit__ch() {} \n\n");
    } else {
        let closest_hit_expr = ir.closest_hit_tokens.as_ref().map(|ts| syn::parse2::<Expr>(ts.clone()).unwrap());
        code.push_str("extern \"C\" __global__ void __closesthit__ch() {\n");
        code.push_str("\tunsigned int hit_primitive_id = optixGetPrimitiveIndex();\n");

        code.push_str("\tPayloadBundle* bundle = params.bundle; (void)sizeof(bundle);\n");

        code.push_str("\t(void)sizeof(bundle); // Suppress NVCC #550\n\n");
        if let Some(expr) = &closest_hit_expr {
            code.push_str(&format!("\t{}\n", translate_expr(&ctx, expr)));
        }
        code.push_str("}\n\n");
    }

    code
}





pub fn compile_to_ptx(cuda_source: &str) -> String {
    use std::env; use std::fs; use std::process::Command;
    let optix_path = env::var("OPTIX_PATH").expect("OPTIX_PATH env variable must be set!");
    let include_arg = format!("-I{}/include", optix_path);
    let temp_dir = env::temp_dir().join("gprt_build"); fs::create_dir_all(&temp_dir).unwrap();
    let cu_file = temp_dir.join("shader.cu"); let ptx_file = temp_dir.join("shader.ptx");
    fs::write(&cu_file, cuda_source).unwrap();
    let arch_arg = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_string());
    let status = Command::new("nvcc").arg("-ptx").arg("-O3").arg("--use_fast_math").arg("-std=c++11").arg(format!("-arch={}", arch_arg)).arg(&include_arg).arg("-o").arg(&ptx_file).arg(&cu_file).status().expect("Failed to run nvcc.");
    if !status.success() { panic!("NVCC compilation failed!"); }
    let ptx_content = fs::read_to_string(&ptx_file).unwrap();
    let _ = fs::remove_file(cu_file); let _ = fs::remove_file(ptx_file);
    ptx_content
}

pub fn compile_to_optixir(cuda_source: &str) -> Vec<u8> {
    use std::env; use std::fs; use std::process::Command;
    let optix_path = env::var("OPTIX_PATH").expect("OPTIX_PATH env variable must be set!");
    let include_arg = format!("-I{}/include", optix_path);
    let temp_dir = env::temp_dir().join("gprt_build"); fs::create_dir_all(&temp_dir).unwrap();
    let cu_file = temp_dir.join("shader.cu"); let ir_file = temp_dir.join("shader.optixir");
    fs::write(&cu_file, cuda_source).unwrap();
    let arch_arg = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_string());
    
    // Compile to OptiX-IR (LLVM Bitcode)
    let status = Command::new("nvcc")
        .arg("-optix-ir").arg("-O3").arg("--use_fast_math").arg("-std=c++11")
        .arg(format!("-arch={}", arch_arg)).arg(&include_arg)
        .arg("-o").arg(&ir_file).arg(&cu_file)
        .status().expect("Failed to run nvcc.");
        
    if !status.success() { panic!("NVCC compilation failed!"); }
    let ir_content = fs::read(&ir_file).unwrap();
    let _ = fs::remove_file(cu_file); let _ = fs::remove_file(ir_file);
    ir_content
}
