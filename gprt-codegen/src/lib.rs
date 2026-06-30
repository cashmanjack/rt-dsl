use gprt_ir::{RtProgram, ShaderNode};
use syn::{Expr, ExprMethodCall, parse_quote, visit::Visit};
use quote::quote;
use std::process::Command;
use std::env;
use std::fs;

// ==========================================
// 1. THE POLYGLOT TRANSPILER (Rust -> CUDA)
// ==========================================
pub struct GpuCodegen {
    pub cuda_string: String,
    pub indent: usize,
}

impl GpuCodegen {
    pub fn new() -> Self { Self { cuda_string: String::new(), indent: 1 } }
    
    fn emit(&mut self, s: &str) {
        for _ in 0..self.indent { self.cuda_string.push_str("\t"); }
        self.cuda_string.push_str(s);
        self.cuda_string.push('\n');
    }
    
    pub fn transpile_expr(&mut self, expr: &Expr) -> String {
        let mut visitor = RustToCudaVisitor { out: String::new() };
        visitor.visit_expr(expr);
        visitor.out
    }
}

struct RustToCudaVisitor { out: String }

impl<'ast> Visit<'ast> for RustToCudaVisitor {
    fn visit_expr_method_call(&mut self, i: &'ast ExprMethodCall) {
        if i.method == "push" {
            self.out.push_str("atomicAdd(&lens[qid], 1); /* push */");
        } else if i.method == "distance" {
            self.out.push_str("fast_distance(");
            self.visit_expr(&i.receiver);
            self.out.push_str(", ");
            self.visit_expr(&i.args[0]);
            self.out.push_str(")");
        } else {
            self.visit_expr(&i.receiver);
            self.out.push_str(".");
            self.out.push_str(&i.method.to_string());
            self.out.push_str("(");
            for (idx, arg) in i.args.iter().enumerate() {
                if idx > 0 { self.out.push_str(", "); }
                self.visit_expr(arg);
            }
            self.out.push_str(")");
        }
    }
    
    fn visit_expr_path(&mut self, i: &'ast syn::ExprPath) {
        self.out.push_str(&quote!(#i).to_string().replace(" ", ""));
    }
    
    fn visit_expr_lit(&mut self, i: &'ast syn::ExprLit) {
        self.out.push_str(&quote!(#i).to_string());
    }
    
    fn visit_expr(&mut self, i: &'ast Expr) {
        if self.out.is_empty() {
            self.out.push_str(&quote!(#i).to_string().replace(" ", ""));
        } else {
            syn::visit::visit_expr(self, i);
        }
    }
}

// ==========================================
// 2. THE AST LOWERING (ShaderNode -> CUDA)
// ==========================================
impl GpuCodegen {
    pub fn lower_node(&mut self, node: &ShaderNode) {
        match node {
            ShaderNode::TraceRay { tmax, on_hit, on_miss } => {
                let tmax_str = quote!(#tmax).to_string();
                self.emit(&format!("optixTrace(params.handle, origin, dir, 0.0f, {}, 0.0f, 255u, OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0u, 1u, 0u, p0, p1);", tmax_str));
                self.emit("if (p1 != 0xFFFFFFFF) {");
                self.indent += 1;
                self.lower_node(on_hit);
                self.indent -= 1;
                self.emit("} else {");
                self.indent += 1;
                self.lower_node(on_miss);
                self.indent -= 1;
                self.emit("}");
            }
            ShaderNode::PushToDynamicArray { array_name, value } => {
                let val_str = self.transpile_expr(&parse_quote!(#value));
                self.emit(&format!("unsigned int __idx = atomicAdd(&bundle->dyn_lens[{}], 1u);", array_name));
                self.emit(&format!("if (__idx < bundle->dyn_caps[{}]) bundle->dyn_ptrs[{}][__idx] = {};", array_name, array_name, val_str));
            }
            ShaderNode::If { condition, then_body, else_body } => {
                let cond_str = self.transpile_expr(&parse_quote!(#condition));
                self.emit(&format!("if ({}) {{", cond_str));
                self.indent += 1;
                self.lower_node(then_body);
                self.indent -= 1;
                self.emit("} else {");
                self.indent += 1;
                self.lower_node(else_body);
                self.indent -= 1;
                self.emit("}");
            }
            ShaderNode::Block(stmts) => {
                for stmt in stmts { self.lower_node(stmt); }
            }
            ShaderNode::RawCuda(code) => {
                self.emit(code);
            }
            _ => {}
        }
    }
}

pub fn compile_program(ir: &RtProgram) -> Vec<u8> {
    let mut codegen = GpuCodegen::new();
    
    codegen.emit("#include <optix.h>");
    codegen.emit("#include <optix_device.h>");
    codegen.emit("#include <vector_types.h>");
    codegen.emit("");
    codegen.emit("struct PayloadBundle { unsigned long long dyn_ptrs[8]; unsigned int dyn_caps[8]; unsigned long long dyn_lens[8]; unsigned long long val_ptrs[8]; };");

    codegen.emit("struct LaunchParams { OptixTraversableHandle handle; float4* rays; int num_rays; PayloadBundle* bundle; float4* geom; };");
    codegen.emit("__constant__ LaunchParams params;");
    
    codegen.emit("");
    codegen.emit("extern \"C\" __global__ void __raygen__rg() {");
    codegen.indent += 1;
    codegen.lower_node(&ir.raygen_body);
    codegen.indent -= 1;
    codegen.emit("}");
    
    if let Some(miss) = &ir.miss_body {
        codegen.emit("extern \"C\" __global__ void __miss__ms() {");
        codegen.indent += 1;
        codegen.lower_node(miss);
        codegen.indent -= 1;
        codegen.emit("}");
    } else {
        codegen.emit("extern \"C\" __global__ void __miss__ms() {}");
    }

    if let Some(anyhit) = &ir.anyhit_body {
        codegen.emit("extern \"C\" __global__ void __anyhit__ah() {");
        codegen.indent += 1;
        codegen.lower_node(anyhit);
        codegen.indent -= 1;
        codegen.emit("}");
    } else {
        codegen.emit("extern \"C\" __global__ void __anyhit__ah() {}");
    }

    if let Some(closesthit) = &ir.closesthit_body {
        codegen.emit("extern \"C\" __global__ void __closesthit__ch() {");
        codegen.indent += 1;
        codegen.lower_node(closesthit);
        codegen.indent -= 1;
        codegen.emit("}");
    } else {
        codegen.emit("extern \"C\" __global__ void __closesthit__ch() { optixSetPayload_0(1); }");
    }

    // FULLY ABSTRACTED: Generates dynamic intersection shaders strictly if defined by macro AST
    if let Some(intersection) = &ir.intersection_body {
        codegen.emit("extern \"C\" __global__ void __intersection__is() {");
        codegen.indent += 1;
        codegen.lower_node(intersection);
        codegen.indent -= 1;
        codegen.emit("}");
    } else {
        codegen.emit("extern \"C\" __global__ void __intersection__is() {}");
    }
    
    compile_program_to_optixir(&codegen.cuda_string)
}

pub fn compile_program_to_optixir(cuda_source: &str) -> Vec<u8> {
    let optix_path = env::var("OPTIX_PATH")
        .unwrap_or_else(|_| option_env!("OPTIX_PATH").unwrap_or("/home/min/a/cashman3/optix_sdk").to_string());
    let include_arg = format!("-I{}/include", optix_path);
    let temp_dir = env::temp_dir().join("gprt_build"); fs::create_dir_all(&temp_dir).unwrap();
    let cu_file = temp_dir.join("shader.cu"); let ir_file = temp_dir.join("shader.optixir");
    fs::write(&cu_file, cuda_source).unwrap();
    let arch_arg = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_string());
    
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
