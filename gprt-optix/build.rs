// gprt-optix/build.rs
use std::env;

fn main() {
    println!("cargo:warning=DEBUG: Starting CMake build...");
    let optix_path = env::var("OPTIX_PATH").unwrap();
    println!("cargo:warning=DEBUG: OPTIX_PATH is {}", optix_path);

    let optix_path = env::var("OPTIX_PATH").expect("OPTIX_PATH must be set");
    
    // Build the C++ Host Library
    let dst = cmake::Config::new("host")
        .define("OPTIX_PATH", &optix_path)
        .build();

    // Link the compiled C++ static library
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=gprt_optix_host");
    
    // === NEW: Tell Rust where to find CUDA ===
    let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-search=native={}/lib", cuda_path);
    
    // Link CUDA and C++ standard libraries
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
