struct SphereSbtData { float4* spheres; char padding[8]; };
extern "C" __global__ void __intersection__is() {
    const SphereSbtData* sbt = (const SphereSbtData*)optixGetSbtDataPointer();
    unsigned int prim_id = optixGetPrimitiveIndex();
    float4 sphere = sbt->spheres[prim_id];
    float3 o = optixGetObjectRayOrigin();
    float3 oc = make_float3(o.x - sphere.x, o.y - sphere.y, o.z - sphere.z);
    float3 dir = optixGetObjectRayDirection();
    float radius = sphere.w;
    float a = gprt_dot(dir, dir); float b = 2.0f * gprt_dot(oc, dir); float c = gprt_dot(oc, oc) - radius * radius;
    float disc = b*b - 4.0f*a*c;
    if (disc > 0.0f) {
        float t = (-b - sqrtf(disc)) / a;
        if (t > optixGetRayTmin() && t < optixGetRayTmax()) optixReportIntersection(t, 0);
        else { t = (-b + sqrtf(disc)) / a; if (t > optixGetRayTmin() && t < optixGetRayTmax()) optixReportIntersection(t, 0); }
    }
}
