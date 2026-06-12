// gprt-optix/host/optix_runner.cpp
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <sstream>
#include <cstring> 

// Official Nvidia definition header providing proper C-linkage for the function table
#include <optix_function_table_definition.h>

// Robust macro utilities for cross-language CUDA/OptiX error tracking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t rc = call; \
        if (rc != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(rc) << " at line " << __LINE__ << std::endl; \
            exit(rc); \
        } \
    } while (0)

#define OPTIX_CHECK(call) \
    do { \
        OptixResult rc = call; \
        if (rc != OPTIX_SUCCESS) { \
            std::cerr << "OptiX Error: " << rc << " at line " << __LINE__ << std::endl; \
            exit(rc); \
        } \
    } while (0)

// Opaque layout definitions matching the Rust FFI boundaries
struct GprtPipeline {
    OptixDeviceContext context;
    OptixModule module;
    OptixPipeline pipeline;
    OptixShaderBindingTable sbt;
    
    // Device memory handles for records inside the SBT
    CUdeviceptr d_raygen_record;
    CUdeviceptr d_miss_record;
    CUdeviceptr d_hitgroup_record;
};

struct GprtBvh {
    OptixTraversableHandle handle;
    CUdeviceptr d_buffer; 
};

// Uniform layout mapping the unified device pointers down to hardware registers
struct LaunchParams {
    OptixTraversableHandle traversable;
    const float* queries;
    unsigned int* neighbors_out;
    unsigned int* neighbors_len;
    unsigned int* closest_out;
};

// Internal structures template for matching SBT alignment requirements
template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<int> EmptyDataRecord;

extern "C" {

// ============================================================================
// 1. STATEFUL PIPELINE INITIALIZATION (Runs Exactly ONCE via OnceLock)
// ============================================================================
GprtPipeline* gprt_pipeline_create(const char* ptx_code) {

    std::cout << "\n=============================================" << std::endl;
    std::cout << "--- DEBUG: RAW PTX PASSED FROM RUST DSL ---" << std::endl;
    std::cout << ptx_code << std::endl;
    std::cout << "=============================================\n" << std::endl;

    GprtPipeline* pipe = new GprtPipeline();

    // Initialize CUDA driver context for primary device GPU
    CUDA_CHECK(cudaFree(0)); 
    
    // Initialize OptiX function table entry points
    OPTIX_CHECK(optixInit());

    // Create the low-level OptiX device execution context
    OptixDeviceContextOptions options = {};
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    OPTIX_CHECK(optixDeviceContextCreate(0, &options, &pipe->context));

    // Compile incoming runtime PTX source into a stateful hardware module
    OptixModuleCompileOptions module_compile_options = {};
    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    
    // Configured to 6 to map your DSL's payload register usage
    pipeline_compile_options.numPayloadValues = 6; 
    pipeline_compile_options.numAttributeValues = 2;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    char log[2048];
    size_t log_size = sizeof(log);
    
    OptixResult res = optixModuleCreate(
        pipe->context,
        &module_compile_options,
        &pipeline_compile_options,
        ptx_code,
        strlen(ptx_code),
        log,
        &log_size,
        &pipe->module
    );

    if (res != OPTIX_SUCCESS) {
        std::cerr << "\n=== OptiX Module Compilation Failed! ===" << std::endl;
        std::cerr << log << std::endl;
        std::cerr << "========================================\n" << std::endl;
        OPTIX_CHECK(res); 
    }

    // Create program execution groups matching your DSL parsed hooks
    OptixProgramGroupOptions pg_options = {};
    std::vector<OptixProgramGroup> program_groups;
    OptixResult pg_res;

    // Raygen group setup
    OptixProgramGroupDesc rg_desc = {};
    rg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rg_desc.raygen.module = pipe->module;
    rg_desc.raygen.entryFunctionName = "__raygen__rg"; 
    OptixProgramGroup rg_pg;
    
    log_size = sizeof(log); // Reset log capacity
    pg_res = optixProgramGroupCreate(pipe->context, &rg_desc, 1, &pg_options, log, &log_size, &rg_pg);
    if (pg_res != OPTIX_SUCCESS) {
        std::cerr << "\n=== Raygen Program Group Creation Failed! ===" << std::endl;
        std::cerr << log << std::endl;
        std::cerr << "=============================================\n" << std::endl;
        OPTIX_CHECK(pg_res);
    }
    program_groups.push_back(rg_pg);

    // Miss group setup
    OptixProgramGroupDesc ms_desc = {};
    ms_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    ms_desc.miss.module = pipe->module;
    ms_desc.miss.entryFunctionName = "__miss__ms"; 
    OptixProgramGroup ms_pg;
    
    log_size = sizeof(log); // Reset log capacity
    pg_res = optixProgramGroupCreate(pipe->context, &ms_desc, 1, &pg_options, log, &log_size, &ms_pg);
    if (pg_res != OPTIX_SUCCESS) {
        std::cerr << "\n=== Miss Program Group Creation Failed! ===" << std::endl;
        std::cerr << log << std::endl;
        std::cerr << "===========================================\n" << std::endl;
        OPTIX_CHECK(pg_res);
    }
    program_groups.push_back(ms_pg);

    // Closest/Any Hit unified group setup
    OptixProgramGroupDesc hg_desc = {};
    hg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hg_desc.hitgroup.moduleCH = pipe->module;
    hg_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch"; 
    hg_desc.hitgroup.moduleAH = pipe->module;
    hg_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah"; 
    OptixProgramGroup hg_pg;
    
    log_size = sizeof(log); // Reset log capacity
    pg_res = optixProgramGroupCreate(pipe->context, &hg_desc, 1, &pg_options, log, &log_size, &hg_pg);
    if (pg_res != OPTIX_SUCCESS) {
        std::cerr << "\n=== Hitgroup Program Group Creation Failed! ===" << std::endl;
        std::cerr << log << std::endl;
        std::cerr << "==============================================\n" << std::endl;
        OPTIX_CHECK(pg_res);
    }
    program_groups.push_back(hg_pg);

    // Link everything into the ultimate execution pipeline
    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 2;
    log_size = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(
        pipe->context,
        &pipeline_compile_options,
        &link_options,
        program_groups.data(),
        program_groups.size(),
        log,
        &log_size,
        &pipe->pipeline
    ));

    // Pack records into the Shader Binding Table (SBT) map layout
    EmptyDataRecord rg_rec, ms_rec, hg_rec;
    OPTIX_CHECK(optixSbtRecordPackHeader(rg_pg, &rg_rec));
    OPTIX_CHECK(optixSbtRecordPackHeader(ms_pg, &ms_rec));
    OPTIX_CHECK(optixSbtRecordPackHeader(hg_pg, &hg_rec));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pipe->d_raygen_record), sizeof(EmptyDataRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(pipe->d_raygen_record), &rg_rec, sizeof(EmptyDataRecord), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pipe->d_miss_record), sizeof(EmptyDataRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(pipe->d_miss_record), &ms_rec, sizeof(EmptyDataRecord), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pipe->d_hitgroup_record), sizeof(EmptyDataRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(pipe->d_hitgroup_record), &hg_rec, sizeof(EmptyDataRecord), cudaMemcpyHostToDevice));

    pipe->sbt.raygenRecord = pipe->d_raygen_record;
    pipe->sbt.missRecordBase = pipe->d_miss_record;
    pipe->sbt.missRecordStrideInBytes = sizeof(EmptyDataRecord);
    pipe->sbt.missRecordCount = 1;
    pipe->sbt.hitgroupRecordBase = pipe->d_hitgroup_record;
    pipe->sbt.hitgroupRecordStrideInBytes = sizeof(EmptyDataRecord);
    pipe->sbt.hitgroupRecordCount = 1;

    return pipe;
}

// ============================================================================
// 2. STATEFUL ACCELERATION STRUCTURE BUILD (Runs Only When Geometry Changes)
// ============================================================================
GprtBvh* gprt_bvh_build(GprtPipeline* pipe, const float* d_geom, int count) {
    GprtBvh* bvh = new GprtBvh();

    uint32_t aabbs_build_flags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    build_input.customPrimitiveArray.aabbBuffers = reinterpret_cast<const CUdeviceptr*>(&d_geom);
    build_input.customPrimitiveArray.numPrimitives = count;
    build_input.customPrimitiveArray.strideInBytes = sizeof(float) * 6; 

    uint32_t num_v_mask = 1;
    build_input.customPrimitiveArray.flags = &num_v_mask;
    build_input.customPrimitiveArray.numSbtRecords = 1;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = aabbs_build_flags;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(pipe->context, &accel_options, &build_input, 1, &buffer_sizes));

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), buffer_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&bvh->d_buffer), buffer_sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        pipe->context,
        0, 
        &accel_options,
        &build_input,
        1, 
        d_temp_buffer,
        buffer_sizes.tempSizeInBytes,
        bvh->d_buffer,
        buffer_sizes.outputSizeInBytes,
        &bvh->handle,
        nullptr,
        0
    ));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
    return bvh;
}

// ============================================================================
// 3. HARDWARE RAY LAUNCH (Runs Every Single Frame/Query Loop)
// ============================================================================
void gprt_launch(
    GprtPipeline* pipe, GprtBvh* bvh, 
    const float* d_queries, int count,
    unsigned int* d_neighbors_out, unsigned int* d_neighbors_len,
    unsigned int* d_closest_out
) {
    LaunchParams params;
    params.traversable = bvh->handle;
    params.queries = d_queries;
    params.neighbors_out = d_neighbors_out;
    params.neighbors_len = d_neighbors_len;
    params.closest_out = d_closest_out;

    CUdeviceptr d_params;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(LaunchParams)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params), &params, sizeof(LaunchParams), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(
        pipe->pipeline,
        0, 
        d_params,
        sizeof(LaunchParams),
        &pipe->sbt,
        count, 
        1,     
        1      
    ));

    CUDA_CHECK(cudaStreamSynchronize(0));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));
}

} // extern "C"
