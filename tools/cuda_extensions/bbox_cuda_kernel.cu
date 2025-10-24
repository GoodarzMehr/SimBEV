// Academic Software License: Copyright © 2025 Goodarz Mehr.

// CUDA kernel for efficiently checking how many points lie within a given 3D
// bounding box.

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t norm3d(scalar_t x, scalar_t y, scalar_t z) {
    return sqrt(x * x + y * y + z * z);
}

template <typename scalar_t>
__global__ void is_inside_bbox_kernel(
    const scalar_t* __restrict__ points,
    const scalar_t* __restrict__ bbox,
    int* __restrict__ count,
    int n_points)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_points) return;
    
    // Load point
    const scalar_t px = points[idx * 3 + 0];
    const scalar_t py = points[idx * 3 + 1];
    const scalar_t pz = points[idx * 3 + 2];
    
    // Reference point p0 = bbox[0]
    const scalar_t p0x = bbox[0];
    const scalar_t p0y = bbox[1];
    const scalar_t p0z = bbox[2];
    
    // Local axes
    // u = bbox[2] - bbox[0]
    const scalar_t ux = bbox[6] - p0x; // bbox[2 * 3 + 0]
    const scalar_t uy = bbox[7] - p0y; // bbox[2 * 3 + 1]
    const scalar_t uz = bbox[8] - p0z; // bbox[2 * 3 + 2]

    // v = bbox[4] - bbox[0]
    const scalar_t vx = bbox[12] - p0x; // bbox[4 * 3 + 0]
    const scalar_t vy = bbox[13] - p0y; // bbox[4 * 3 + 1]
    const scalar_t vz = bbox[14] - p0z; // bbox[4 * 3 + 2]
    
    // w = bbox[1] - bbox[0]
    const scalar_t wx = bbox[3] - p0x; // bbox[1 * 3 + 0]
    const scalar_t wy = bbox[4] - p0y; // bbox[1 * 3 + 1]
    const scalar_t wz = bbox[5] - p0z; // bbox[1 * 3 + 2]

    // Compute lengths, compare against epsilon to avoid division by zero
    const scalar_t epsilon = scalar_t(1e-8);
    const scalar_t u_len = fmax(norm3d(ux, uy, uz), epsilon);
    const scalar_t v_len = fmax(norm3d(vx, vy, vz), epsilon);
    const scalar_t w_len = fmax(norm3d(wx, wy, wz), epsilon);
    
    // Normalize
    const scalar_t u_nx = ux / u_len;
    const scalar_t u_ny = uy / u_len;
    const scalar_t u_nz = uz / u_len;
    
    const scalar_t v_nx = vx / v_len;
    const scalar_t v_ny = vy / v_len;
    const scalar_t v_nz = vz / v_len;
    
    const scalar_t w_nx = wx / w_len;
    const scalar_t w_ny = wy / w_len;
    const scalar_t w_nz = wz / w_len;
    
    // Translate point
    const scalar_t pt_x = px - p0x;
    const scalar_t pt_y = py - p0y;
    const scalar_t pt_z = pz - p0z;
    
    // Project onto local axes (dot products)
    const scalar_t proj_u = pt_x * u_nx + pt_y * u_ny + pt_z * u_nz;
    const scalar_t proj_v = pt_x * v_nx + pt_y * v_ny + pt_z * v_nz;
    const scalar_t proj_w = pt_x * w_nx + pt_y * w_ny + pt_z * w_nz;
    
    // Check bounds and atomically increment count if inside
    bool inside = (proj_u >= scalar_t(0.0) && proj_u <= u_len &&
                   proj_v >= scalar_t(0.0) && proj_v <= v_len &&
                   proj_w >= scalar_t(0.0) && proj_w <= w_len);
    
    if (inside) {
        atomicAdd(count, 1);
    }
}

torch::Tensor is_inside_bbox_cuda(torch::Tensor points, torch::Tensor bbox)
{
    const int n_points = points.size(0);
    
    if (n_points == 0) {
        return torch::zeros({1}, torch::dtype(torch::kInt32).device(points.device()));
    }
    
    // Allocate a single integer for the count
    auto count = torch::zeros({1}, torch::dtype(torch::kInt32).device(points.device()));
    
    const int threads = 256;
    const int blocks = (n_points + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(points.scalar_type(), "is_inside_bbox_cuda", ([&] {
        is_inside_bbox_kernel<scalar_t><<<blocks, threads>>>(
            points.data_ptr<scalar_t>(),
            bbox.data_ptr<scalar_t>(),
            count.data_ptr<int>(),
            n_points
        );
    }));
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel failed: ", cudaGetErrorString(err));
    }
    
    return count;
}