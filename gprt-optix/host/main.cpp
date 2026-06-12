#define OPTIX_STUBS_IMPLEMENTATION
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cstring> // FIX 1: Added for memset
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n"; exit(1); } } while (0)
#define OPTIX_CHECK(call) do { OptixResult res = call; if (res != OPTIX_SUCCESS) { std::cerr << "Optix Error: " << res << "\n"; exit(1); } } while (0)

struct PayloadBundle { unsigned long long dyn_ptrs[8]; unsigned int dyn_caps[8]; unsigned long long dyn_lens[8]; unsigned long long val_ptrs[8]; };


struct LaunchParams { OptixTraversableHandle handle; float4* rays; int num_rays; PayloadBundle* bundle; };

struct SphereSbtData { float4* spheres; char padding[8]; };
struct TriangleSbtData { float3* vertices; unsigned int* indices; };

__align__(OPTIX_SBT_RECORD_ALIGNMENT) struct RaygenSbtRecord { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };
__align__(OPTIX_SBT_RECORD_ALIGNMENT) struct MissSbtRecord { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };
__align__(OPTIX_SBT_RECORD_ALIGNMENT) struct HitGroupSbtRecord { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; char data[32]; };

struct CGprtPipeline {
    OptixDeviceContext context; OptixModule module; OptixPipeline pipeline; OptixShaderBindingTable sbt;
    CUdeviceptr d_raygen, d_miss, d_hitgroup;
    PayloadBundle* h_bundle; CUdeviceptr d_bundle;
    std::map<std::string, unsigned int*> d_arrays; std::map<std::string, unsigned int*> d_array_lens; std::map<std::string, unsigned int*> d_values;
    int next_array_idx = 0; int next_val_idx = 0;

    CUdeviceptr d_rays;
    CUdeviceptr d_params;
    int max_rays_capacity;
};

// Updated to support both Spheres and Triangles
struct CGprtBvh { OptixTraversableHandle handle; CUdeviceptr d_geom, d_aabbs, d_indices, d_gasOutput; size_t gasOutputSize; };

extern "C" {

CGprtPipeline* gprt_pipeline_create(const void* ir_data, size_t ir_size, int is_hardware_triangle) {
    CUDA_CHECK(cudaFree(0)); OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {}; options.logCallbackLevel = 4;
    CGprtPipeline* pipe = new CGprtPipeline();
    OPTIX_CHECK(optixDeviceContextCreate(0, &options, &pipe->context));
    pipe->h_bundle = new PayloadBundle(); memset(pipe->h_bundle, 0, sizeof(PayloadBundle));
    CUDA_CHECK(cudaMalloc((void**)&pipe->d_bundle, sizeof(PayloadBundle)));

    OptixModuleCompileOptions modOpts = {}; OptixPipelineCompileOptions pipeOpts = {};
    pipeOpts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS; 
    pipeOpts.numPayloadValues = 2; pipeOpts.pipelineLaunchParamsVariableName = "params";
    
    // FIX 2: Cast ir_data to const char* for OptiX API compatibility
    OPTIX_CHECK(optixModuleCreate(pipe->context, &modOpts, &pipeOpts, (const char*)ir_data, ir_size, nullptr, nullptr, &pipe->module));

    OptixProgramGroupOptions pgOpts = {}; std::vector<OptixProgramGroup> pgs(3);
    OptixProgramGroupDesc rgDesc = {}; rgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN; rgDesc.raygen.module = pipe->module; rgDesc.raygen.entryFunctionName = "__raygen__rg";
    OptixProgramGroupDesc msDesc = {}; msDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS; msDesc.miss.module = pipe->module; msDesc.miss.entryFunctionName = "__miss__ms";
    OptixProgramGroupDesc hgDesc = {}; hgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP; 
    
    if (is_hardware_triangle) {
        hgDesc.hitgroup.moduleIS = nullptr; hgDesc.hitgroup.entryFunctionNameIS = nullptr; 
    } else {
        hgDesc.hitgroup.moduleIS = pipe->module; hgDesc.hitgroup.entryFunctionNameIS = "__intersection__is"; 
    }
    hgDesc.hitgroup.moduleAH = pipe->module; hgDesc.hitgroup.entryFunctionNameAH = "__anyhit__ah"; 
    hgDesc.hitgroup.moduleCH = pipe->module; hgDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    
    OPTIX_CHECK(optixProgramGroupCreate(pipe->context, &rgDesc, 1, &pgOpts, nullptr, nullptr, &pgs[0]));
    OPTIX_CHECK(optixProgramGroupCreate(pipe->context, &msDesc, 1, &pgOpts, nullptr, nullptr, &pgs[1]));
    OPTIX_CHECK(optixProgramGroupCreate(pipe->context, &hgDesc, 1, &pgOpts, nullptr, nullptr, &pgs[2]));

    OptixPipelineLinkOptions linkOpts = {}; linkOpts.maxTraceDepth = 1;
    OPTIX_CHECK(optixPipelineCreate(pipe->context, &pipeOpts, &linkOpts, pgs.data(), 3, nullptr, nullptr, &pipe->pipeline));

    void *d_rg, *d_ms, *d_hg;
    CUDA_CHECK(cudaMalloc(&d_rg, sizeof(RaygenSbtRecord))); CUDA_CHECK(cudaMalloc(&d_ms, sizeof(MissSbtRecord))); CUDA_CHECK(cudaMalloc(&d_hg, sizeof(HitGroupSbtRecord)));
    pipe->d_raygen = (CUdeviceptr)d_rg; pipe->d_miss = (CUdeviceptr)d_ms; pipe->d_hitgroup = (CUdeviceptr)d_hg;

    RaygenSbtRecord rgSbt; MissSbtRecord msSbt; HitGroupSbtRecord hgSbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(pgs[0], &rgSbt.header)); OPTIX_CHECK(optixSbtRecordPackHeader(pgs[1], &msSbt.header)); OPTIX_CHECK(optixSbtRecordPackHeader(pgs[2], &hgSbt.header));
    CUDA_CHECK(cudaMemcpy(d_rg, &rgSbt, sizeof(RaygenSbtRecord), cudaMemcpyHostToDevice)); 
    CUDA_CHECK(cudaMemcpy(d_ms, &msSbt, sizeof(MissSbtRecord), cudaMemcpyHostToDevice)); 
    CUDA_CHECK(cudaMemcpy(d_hg, &hgSbt, sizeof(HitGroupSbtRecord), cudaMemcpyHostToDevice));

    pipe->sbt = {}; pipe->sbt.raygenRecord = pipe->d_raygen; pipe->sbt.missRecordBase = pipe->d_miss; pipe->sbt.missRecordStrideInBytes = sizeof(MissSbtRecord); pipe->sbt.missRecordCount = 1;
    pipe->sbt.hitgroupRecordBase = pipe->d_hitgroup; pipe->sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord); pipe->sbt.hitgroupRecordCount = 1;
    
    // NEW: Pre-allocate for up to 2 Million rays (32MB)
    pipe->max_rays_capacity = 2000000; 
    
    void* temp_rays_ptr;
    CUDA_CHECK(cudaMalloc(&temp_rays_ptr, pipe->max_rays_capacity * sizeof(float4)));
    pipe->d_rays = (CUdeviceptr)temp_rays_ptr;
    
    void* temp_params_ptr;
    CUDA_CHECK(cudaMalloc(&temp_params_ptr, sizeof(LaunchParams)));
    pipe->d_params = (CUdeviceptr)temp_params_ptr;

    return pipe;
}

void gprt_register_array(CGprtPipeline* pipe, const char* name, int capacity_per_query, int num_queries) {
    if (pipe->d_arrays.count(name)) return;
    int idx = pipe->next_array_idx++;
    
    unsigned int *d_ptr, *d_len;
    size_t data_size = (size_t)capacity_per_query * num_queries * sizeof(unsigned int);
    size_t len_size = (size_t)num_queries * sizeof(unsigned int);
    
    CUDA_CHECK(cudaMalloc(&d_ptr, data_size)); CUDA_CHECK(cudaMemset(d_ptr, 0, data_size));
    CUDA_CHECK(cudaMalloc(&d_len, len_size));  CUDA_CHECK(cudaMemset(d_len, 0, len_size));
    
    pipe->d_arrays[name] = d_ptr; 
    pipe->d_array_lens[name] = d_len;
    pipe->h_bundle->dyn_ptrs[idx] = (unsigned long long)d_ptr;
    pipe->h_bundle->dyn_caps[idx] = (unsigned int)capacity_per_query;
    pipe->h_bundle->dyn_lens[idx] = (unsigned long long)d_len;
}

void gprt_register_value(CGprtPipeline* pipe, const char* name) {
    if (pipe->d_values.count(name)) return;
    int idx = pipe->next_val_idx++;
    unsigned int* d_val; CUDA_CHECK(cudaMalloc(&d_val, sizeof(unsigned int))); CUDA_CHECK(cudaMemset(d_val, 0xFF, sizeof(unsigned int)));
    pipe->d_values[name] = d_val; pipe->h_bundle->val_ptrs[idx] = (unsigned long long)d_val;
}







CGprtBvh* gprt_bvh_build(CGprtPipeline* pipe, const float* h_geom, const float* h_aabbs, int count, int geom_bytes) {
    CGprtBvh* bvh = new CGprtBvh();
    void *d_sp, *d_ab;
    CUDA_CHECK(cudaMalloc(&d_sp, geom_bytes));
    CUDA_CHECK(cudaMemcpy(d_sp, h_geom, geom_bytes, cudaMemcpyHostToDevice));
    bvh->d_geom = (CUdeviceptr)d_sp;

    CUDA_CHECK(cudaMalloc(&d_ab, count * sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(d_ab, h_aabbs, count * sizeof(OptixAabb), cudaMemcpyHostToDevice));
    bvh->d_aabbs = (CUdeviceptr)d_ab;

    OptixBuildInput bi = {};
    bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    bi.customPrimitiveArray.aabbBuffers = &bvh->d_aabbs;
    bi.customPrimitiveArray.numPrimitives = count;
    bi.customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
    unsigned int flag = OPTIX_GEOMETRY_FLAG_NONE;
    bi.customPrimitiveArray.flags = &flag;
    bi.customPrimitiveArray.numSbtRecords = 1;

    // 1. Setup build options with COMPACTION flag
    OptixAccelBuildOptions ao = {};
    ao.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    ao.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes bs;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(pipe->context, &ao, &bi, 1, &bs));

    void *d_t, *d_o;
    CUDA_CHECK(cudaMalloc(&d_t, bs.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(&d_o, bs.outputSizeInBytes));

    // 2. Setup Compaction Size Emission
    OptixAccelEmitDesc emitDesc = {};
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    size_t compactedSize;
    void* d_compactedSize;
    CUDA_CHECK(cudaMalloc(&d_compactedSize, sizeof(size_t)));
    emitDesc.result = (CUdeviceptr)d_compactedSize;

    // 3. Build BVH
    OPTIX_CHECK(optixAccelBuild(
        pipe->context, 0, &ao, &bi, 1,
        (CUdeviceptr)d_t, bs.tempSizeInBytes,
        (CUdeviceptr)d_o, bs.outputSizeInBytes,
        &bvh->handle,
        &emitDesc, 1
    ));

    // 4. Retrieve Compacted Size & Free Temp Buffers
    CUDA_CHECK(cudaMemcpy(&compactedSize, d_compactedSize, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_compactedSize));
    CUDA_CHECK(cudaFree(d_t));

    // 5. Compact the BVH if it saves space
    if (compactedSize < bs.outputSizeInBytes) {
        void* d_compacted;
        CUDA_CHECK(cudaMalloc(&d_compacted, compactedSize));
        OPTIX_CHECK(optixAccelCompact(
            pipe->context, 0,
            bvh->handle,
            (CUdeviceptr)d_compacted, compactedSize,
            &bvh->handle
        ));
        CUDA_CHECK(cudaFree(d_o));
        d_o = d_compacted;
        bs.outputSizeInBytes = compactedSize;
    }

    bvh->d_gasOutput = (CUdeviceptr)d_o;
    bvh->gasOutputSize = bs.outputSizeInBytes;

    SphereSbtData sbtData;
    sbtData.spheres = (float4*)bvh->d_geom;
    CUDA_CHECK(cudaMemcpy((void*)(pipe->d_hitgroup + OPTIX_SBT_RECORD_HEADER_SIZE), &sbtData, sizeof(SphereSbtData), cudaMemcpyHostToDevice));

    return bvh;
}

CGprtBvh* gprt_bvh_build_triangles(CGprtPipeline* pipe, const float* h_verts, const unsigned int* h_indices, int num_triangles, const float* h_aabbs) {
    CGprtBvh* bvh = new CGprtBvh();
    void *d_v, *d_i, *d_ab;
    CUDA_CHECK(cudaMalloc(&d_v, num_triangles * 9 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_v, h_verts, num_triangles * 9 * sizeof(float), cudaMemcpyHostToDevice));
    bvh->d_geom = (CUdeviceptr)d_v;

    CUDA_CHECK(cudaMalloc(&d_i, num_triangles * 3 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_i, h_indices, num_triangles * 3 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    bvh->d_indices = (CUdeviceptr)d_i;

    CUDA_CHECK(cudaMalloc(&d_ab, num_triangles * sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(d_ab, h_aabbs, num_triangles * sizeof(OptixAabb), cudaMemcpyHostToDevice));
    bvh->d_aabbs = (CUdeviceptr)d_ab;

    OptixBuildInput bi = {};
    bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    bi.triangleArray.vertexBuffers = &bvh->d_geom;
    bi.triangleArray.numVertices = num_triangles * 3;
    bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    bi.triangleArray.vertexStrideInBytes = sizeof(float3);
    bi.triangleArray.indexBuffer = bvh->d_indices;
    bi.triangleArray.numIndexTriplets = num_triangles;
    bi.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    bi.triangleArray.indexStrideInBytes = sizeof(unsigned int) * 3;

    unsigned int flag = OPTIX_GEOMETRY_FLAG_NONE;
    bi.triangleArray.flags = &flag;
    bi.triangleArray.numSbtRecords = 1;

    // 1. Setup build options with COMPACTION flag
    OptixAccelBuildOptions ao = {};
    ao.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    ao.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes bs;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(pipe->context, &ao, &bi, 1, &bs));

    void *d_t, *d_o;
    CUDA_CHECK(cudaMalloc(&d_t, bs.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(&d_o, bs.outputSizeInBytes));

    // 2. Setup Compaction Size Emission
    OptixAccelEmitDesc emitDesc = {};
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    size_t compactedSize;
    void* d_compactedSize;
    CUDA_CHECK(cudaMalloc(&d_compactedSize, sizeof(size_t)));
    emitDesc.result = (CUdeviceptr)d_compactedSize;

    // 3. Build BVH
    OPTIX_CHECK(optixAccelBuild(
        pipe->context, 0, &ao, &bi, 1,
        (CUdeviceptr)d_t, bs.tempSizeInBytes,
        (CUdeviceptr)d_o, bs.outputSizeInBytes,
        &bvh->handle,
        &emitDesc, 1
    ));

    // 4. Retrieve Compacted Size & Free Temp Buffers
    CUDA_CHECK(cudaMemcpy(&compactedSize, d_compactedSize, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_compactedSize));
    CUDA_CHECK(cudaFree(d_t));

    // 5. Compact the BVH if it saves space
    if (compactedSize < bs.outputSizeInBytes) {
        void* d_compacted;
        CUDA_CHECK(cudaMalloc(&d_compacted, compactedSize));
        OPTIX_CHECK(optixAccelCompact(
            pipe->context, 0,
            bvh->handle,
            (CUdeviceptr)d_compacted, compactedSize,
            &bvh->handle
        ));
        CUDA_CHECK(cudaFree(d_o));
        d_o = d_compacted;
        bs.outputSizeInBytes = compactedSize;
    }

    bvh->d_gasOutput = (CUdeviceptr)d_o;
    bvh->gasOutputSize = bs.outputSizeInBytes;

    TriangleSbtData sbtData;
    sbtData.vertices = (float3*)bvh->d_geom;
    sbtData.indices = (unsigned int*)bvh->d_indices;
    CUDA_CHECK(cudaMemcpy((void*)(pipe->d_hitgroup + OPTIX_SBT_RECORD_HEADER_SIZE), &sbtData, sizeof(TriangleSbtData), cudaMemcpyHostToDevice));

    return bvh;
}








void gprt_bvh_refit(CGprtPipeline* pipe, CGprtBvh* bvh, const float* h_aabbs, int count) {
    CUDA_CHECK(cudaMemcpy((void*)bvh->d_aabbs, h_aabbs, count * sizeof(OptixAabb), cudaMemcpyHostToDevice));
    
    OptixBuildInput bi = {}; 
    bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES; 
    bi.customPrimitiveArray.aabbBuffers = &bvh->d_aabbs; 
    bi.customPrimitiveArray.numPrimitives = count; 
    bi.customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
    unsigned int flag = OPTIX_GEOMETRY_FLAG_NONE; 
    bi.customPrimitiveArray.flags = &flag; 
    bi.customPrimitiveArray.numSbtRecords = 1;
    
    OptixAccelBuildOptions ao = {}; 
    ao.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE; 
    ao.operation = OPTIX_BUILD_OPERATION_UPDATE; 
    
    OptixAccelBufferSizes bs;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(pipe->context, &ao, &bi, 1, &bs));
    
    void* d_temp = 0;
    if (bs.tempSizeInBytes > 0) CUDA_CHECK(cudaMalloc(&d_temp, bs.tempSizeInBytes));
    
    OPTIX_CHECK(optixAccelBuild(pipe->context, 0, &ao, &bi, 1, (CUdeviceptr)d_temp, bs.tempSizeInBytes, bvh->d_gasOutput, bvh->gasOutputSize, &bvh->handle, nullptr, 0));
    
    if (d_temp) CUDA_CHECK(cudaFree(d_temp));
}






void gprt_execute(CGprtPipeline* pipe, CGprtBvh* bvh, const float* h_q, int count) {
    // Fallback realloc if a batch somehow exceeds 2M rays
    if (count > pipe->max_rays_capacity) {
        CUDA_CHECK(cudaFree((void*)pipe->d_rays));
        pipe->max_rays_capacity = count * 2;
        
        void* temp_rays_ptr;
        CUDA_CHECK(cudaMalloc(&temp_rays_ptr, pipe->max_rays_capacity * sizeof(float4)));
        pipe->d_rays = (CUdeviceptr)temp_rays_ptr;
    }
    
    // 1. Copy rays into the PERSISTENT GPU buffer (No cudaMalloc!)
    CUDA_CHECK(cudaMemcpy((void*)pipe->d_rays, h_q, count * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)pipe->d_bundle, pipe->h_bundle, sizeof(PayloadBundle), cudaMemcpyHostToDevice));
    
    // 2. Setup LaunchParams
    LaunchParams p; 
    p.handle = bvh->handle; 
    p.rays = (float4*)pipe->d_rays; 
    p.num_rays = count; 
    p.bundle = (PayloadBundle*)pipe->d_bundle;
    
    // 3. Copy params into PERSISTENT GPU buffer
    CUDA_CHECK(cudaMemcpy((void*)pipe->d_params, &p, sizeof(LaunchParams), cudaMemcpyHostToDevice));
    
    // 4. Launch
    OPTIX_CHECK(optixLaunch(pipe->pipeline, 0, pipe->d_params, sizeof(LaunchParams), &pipe->sbt, count, 1, 1)); 
    CUDA_CHECK(cudaDeviceSynchronize());
}






void gprt_retrieve_array_lengths(CGprtPipeline* pipe, const char* name, unsigned int* h_lengths, int num_queries) {
    if (pipe->d_array_lens.count(name)) {
        CUDA_CHECK(cudaMemcpy(h_lengths, pipe->d_array_lens[name], num_queries * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(pipe->d_array_lens[name], 0, num_queries * sizeof(unsigned int)));
    }
}

void gprt_retrieve_array_flat(CGprtPipeline* pipe, const char* name, unsigned int* h_out, int total_elements) {
    if (pipe->d_arrays.count(name)) {
        CUDA_CHECK(cudaMemcpy(h_out, pipe->d_arrays[name], total_elements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }
}

void gprt_retrieve_value(CGprtPipeline* pipe, const char* name, unsigned int* h_out) {
    if (pipe->d_values.count(name)) CUDA_CHECK(cudaMemcpy(h_out, pipe->d_values[name], sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void gprt_bvh_destroy(CGprtBvh* bvh) {
    if (bvh) {
        if (bvh->d_aabbs) CUDA_CHECK(cudaFree((void*)bvh->d_aabbs));
        // FIX 4: Changed from d_spheres to d_geom and added d_indices
        if (bvh->d_geom) CUDA_CHECK(cudaFree((void*)bvh->d_geom));
        if (bvh->d_indices) CUDA_CHECK(cudaFree((void*)bvh->d_indices));
        if (bvh->d_gasOutput) CUDA_CHECK(cudaFree((void*)bvh->d_gasOutput));
        delete bvh;
    }
}

void gprt_pipeline_destroy(CGprtPipeline* pipe) {
    if (pipe) {
        for (auto& pair : pipe->d_arrays) { if (pair.second) CUDA_CHECK(cudaFree(pair.second)); }
        for (auto& pair : pipe->d_array_lens) { if (pair.second) CUDA_CHECK(cudaFree(pair.second)); }
        for (auto& pair : pipe->d_values) { if (pair.second) CUDA_CHECK(cudaFree(pair.second)); }
        if (pipe->d_raygen) CUDA_CHECK(cudaFree((void*)pipe->d_raygen));
        if (pipe->d_miss) CUDA_CHECK(cudaFree((void*)pipe->d_miss));
        if (pipe->d_hitgroup) CUDA_CHECK(cudaFree((void*)pipe->d_hitgroup));
        if (pipe->d_bundle) CUDA_CHECK(cudaFree((void*)pipe->d_bundle));
        if (pipe->h_bundle) delete pipe->h_bundle;

    if (pipe->d_rays) CUDA_CHECK(cudaFree((void*)pipe->d_rays));
    if (pipe->d_params) CUDA_CHECK(cudaFree((void*)pipe->d_params));

        delete pipe;
    }
}

} // extern "C"
