#include <cuda_runtime.h>

extern "C" __global__ void compute_sphere_aabbs_kernel(const float4* geom, float* aabbs, float radius, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    float4 s = geom[idx];
    int base = idx * 6;
    aabbs[base + 0] = s.x - radius; aabbs[base + 1] = s.y - radius; aabbs[base + 2] = s.z - radius;
    aabbs[base + 3] = s.x + radius; aabbs[base + 4] = s.y + radius; aabbs[base + 5] = s.z + radius;
}

extern "C" void launch_compute_sphere_aabbs(const float4* d_geom, float* d_aabbs, float radius, int count, cudaStream_t stream) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    compute_sphere_aabbs_kernel<<<blocks, threads, 0, stream>>>(d_geom, d_aabbs, radius, count);
}
