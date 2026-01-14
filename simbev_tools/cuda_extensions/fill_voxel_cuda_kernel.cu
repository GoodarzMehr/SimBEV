// Academic Software License: Copyright © 2026 Goodarz Mehr.

// CUDA kernels for filling hollow interior voxels and morphological operations.

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Helper to convert 3D index to linear index.
__device__ __forceinline__ int idx3d(int x, int y, int z, int dim_y, int dim_z) {
    return x * dim_y * dim_z + y * dim_z + z;
}

// Kernel to check if a chunk contains both target class and empty voxels.
// Returns a boolean array indicating which chunks need processing.
__global__ void check_chunks_kernel(
    const uint8_t* __restrict__ input,
    bool* __restrict__ chunk_has_data,
    int dim_x,
    int dim_y,
    int dim_z,
    int chunk_size_x,
    int chunk_size_y,
    int chunk_size_z,
    int num_chunks_x,
    int num_chunks_y,
    int num_chunks_z,
    int target_class
)
{
    const int chunk_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int chunk_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int chunk_z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (chunk_x >= num_chunks_x || chunk_y >= num_chunks_y || chunk_z >= num_chunks_z) return;
    
    const int chunk_idx = chunk_x * num_chunks_y * num_chunks_z + chunk_y * num_chunks_z + chunk_z;
    
    // Calculate chunk boundaries.
    const int x_start = chunk_x * chunk_size_x;
    const int y_start = chunk_y * chunk_size_y;
    const int z_start = chunk_z * chunk_size_z;
    
    const int x_end = min(x_start + chunk_size_x, dim_x);
    const int y_end = min(y_start + chunk_size_y, dim_y);
    const int z_end = min(z_start + chunk_size_z, dim_z);
    
    // Check if chunk contains both the target class and empty voxels.
    bool has_target = false;
    bool has_empty = false;
    
    for (int x = x_start; x < x_end && !(has_target && has_empty); x++) {
        for (int y = y_start; y < y_end && !(has_target && has_empty); y++) {
            for (int z = z_start; z < z_end && !(has_target && has_empty); z++) {
                const int idx = idx3d(x, y, z, dim_y, dim_z);
                
                const uint8_t val = input[idx];
                
                if (val == target_class) has_target = true;
                if (val == 0) has_empty = true;
            }
        }
    }
    
    // Mark chunk to be processed if it has both the target class and empty
    // voxels.
    chunk_has_data[chunk_idx] = has_target && has_empty;
}

// Ray casting kernel: for each empty voxel in an active chunk, cast rays in 6
// directions. Fill the voxel if all 6 rays hit the same target class or the
// top and/or bottom rays go out of bounds.
__global__ void fill_hollow_chunked_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const int* __restrict__ priority_map,
    const bool* __restrict__ active_chunks,
    int dim_x,
    int dim_y,
    int dim_z,
    int chunk_size_x,
    int chunk_size_y,
    int chunk_size_z,
    int num_chunks_x,
    int num_chunks_y,
    int num_chunks_z,
    int target_class,
    int target_priority
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= dim_x || y >= dim_y || z >= dim_z) return;
    
    // Determine which chunk this voxel belongs to.
    const int chunk_x = x / chunk_size_x;
    const int chunk_y = y / chunk_size_y;
    const int chunk_z = z / chunk_size_z;
    
    const int chunk_idx = chunk_x * num_chunks_y * num_chunks_z + chunk_y * num_chunks_z + chunk_z;
    
    const int idx = idx3d(x, y, z, dim_y, dim_z);
    
    const uint8_t current_val = input[idx];
    
    // Copy input to output first.
    output[idx] = current_val;

    // Skip if chunk is not active.
    if (!active_chunks[chunk_idx]) return;
    
    // Only process empty voxels (class 0).
    if (current_val != 0) return;
    
    // Direction vectors: +X, -X, +Y, -Y, +Z, -Z.
    const int dx[6] = {1, -1, 0, 0, 0, 0};
    const int dy[6] = {0, 0, 1, -1, 0, 0};
    const int dz[6] = {0, 0, 0, 0, 1, -1};
    
    int hit_count = 0;
    
    // Cast rays in all 6 directions.
    for (int dir = 0; dir < 6; dir++) {
        int cx = x + dx[dir];
        int cy = y + dy[dir];
        int cz = z + dz[dir];
        
        bool found_target = false;
        
        // March until hitting a voxel of the same class, or going out of
        // bounds.
        while (cx >= 0 && cx < dim_x && cy >= 0 && cy < dim_y && cz >= 0 && cz < dim_z)
        {
            const int cidx = idx3d(cx, cy, cz, dim_y, dim_z);
            
            const uint8_t cval = input[cidx];
            
            if (cval != 0) {
                if (cval == target_class) {
                    found_target = true;
                }
                
                break;
            }
            
            cx += dx[dir];
            cy += dy[dir];
            cz += dz[dir];
        }
        
        if (found_target) {
            hit_count++;
        }
    }
    
    if (hit_count == 6)
    {
        output[idx] = static_cast<uint8_t>(target_class);
    }
}

// 3D dilation kernel for morphological closing.
__global__ void dilate_3d_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int dim_x,
    int dim_y,
    int dim_z,
    int target_class,
    int radius
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= dim_x || y >= dim_y || z >= dim_z) return;
    
    const int idx = idx3d(x, y, z, dim_y, dim_z);
    
    const uint8_t current_val = input[idx];
    
    // Copy input to output first.
    output[idx] = current_val;
    
    // Only dilate into empty voxels.
    if (current_val != 0) return;
    
    // Check neighborhood for target class.
    for (int dx = -radius; dx <= radius; dx++) {
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dz = -radius; dz <= radius; dz++) {
                const int nx = x + dx;
                const int ny = y + dy;
                const int nz = z + dz;
                
                if (nx >= 0 && nx < dim_x && ny >= 0 && ny < dim_y && nz >= 0 && nz < dim_z)
                {
                    const int nidx = idx3d(nx, ny, nz, dim_y, dim_z);
                    
                    if (input[nidx] == target_class) {
                        output[idx] = static_cast<uint8_t>(target_class);
                        
                        return;
                    }
                }
            }
        }
    }
}

// 3D erosion kernel for morphological closing. Only erodes voxels that were
// added during dilation.
__global__ void erode_3d_kernel(
    const uint8_t* __restrict__ original,
    const uint8_t* __restrict__ dilated,
    uint8_t* __restrict__ output,
    int dim_x,
    int dim_y,
    int dim_z,
    int target_class,
    int radius
)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= dim_x || y >= dim_y || z >= dim_z) return;
    
    const int idx = idx3d(x, y, z, dim_y, dim_z);
    
    const uint8_t dilated_val = dilated[idx];
    const uint8_t original_val = original[idx];
    
    // Copy the dilated value to output first.
    output[idx] = dilated_val;
    
    // Only consider voxels that are the target class in the dilated grid.
    if (dilated_val != target_class) return;
    
    // If it was already the target class in the original grid, keep it.
    if (original_val == target_class) return;
    
    // This voxel was added by dilation. Check if it should be eroded. Erode
    // if any neighbor in the radius is not the target class in the dilated
    // grid.
    for (int dx = -radius; dx <= radius; dx++) {
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dz = -radius; dz <= radius; dz++) {
                const int nx = x + dx;
                const int ny = y + dy;
                const int nz = z + dz;
                
                // Boundary does not count as target class.
                if (nx < 0 || nx >= dim_x || ny < 0 || ny >= dim_y || nz < 0 || nz >= dim_z)
                {
                    output[idx] = 0;

                    return;
                }
                
                const int nidx = idx3d(nx, ny, nz, dim_y, dim_z);
                if (dilated[nidx] != target_class) {
                    output[idx] = 0;
                    
                    return;
                }
            }
        }
    }
}

// Launch configuration helper.
dim3 get_block_size() {
    return dim3(8, 8, 8);
}

dim3 get_grid_size(int dim_x, int dim_y, int dim_z, dim3 block) {
    return dim3(
        (dim_x + block.x - 1) / block.x,
        (dim_y + block.y - 1) / block.y,
        (dim_z + block.z - 1) / block.z
    );
}

torch::Tensor fill_hollow_voxels_cuda(
    torch::Tensor voxel_grid,
    torch::Tensor priority_map,
    int target_class,
    int chunk_size_x,
    int chunk_size_y,
    int chunk_size_z
)
{
    const int dim_x = voxel_grid.size(0);
    const int dim_y = voxel_grid.size(1);
    const int dim_z = voxel_grid.size(2);
    
    // Calculate the number of chunks.
    const int num_chunks_x = (dim_x + chunk_size_x - 1) / chunk_size_x;
    const int num_chunks_y = (dim_y + chunk_size_y - 1) / chunk_size_y;
    const int num_chunks_z = (dim_z + chunk_size_z - 1) / chunk_size_z;
    
    const int num_chunks = num_chunks_x * num_chunks_y * num_chunks_z;
    
    // Get target priority.
    auto priority_cpu = priority_map.cpu();
    
    const int target_priority = priority_cpu.data_ptr<int>()[target_class];
    
    // Allocate chunk activity array.
    auto options = torch::TensorOptions().dtype(torch::kBool).device(voxel_grid.device());
    auto active_chunks = torch::zeros({num_chunks}, options);
    
    // Identify active chunks (containing both target class and empty voxels).
    dim3 chunk_block(8, 8, 8);
    
    dim3 chunk_grid(
        (num_chunks_x + chunk_block.x - 1) / chunk_block.x,
        (num_chunks_y + chunk_block.y - 1) / chunk_block.y,
        (num_chunks_z + chunk_block.z - 1) / chunk_block.z
    );
    
    check_chunks_kernel<<<chunk_grid, chunk_block>>>(
        voxel_grid.data_ptr<uint8_t>(),
        active_chunks.data_ptr<bool>(),
        dim_x,
        dim_y,
        dim_z,
        chunk_size_x,
        chunk_size_y,
        chunk_size_z,
        num_chunks_x,
        num_chunks_y,
        num_chunks_z,
        target_class
    );
    
    cudaError_t err = cudaGetLastError();
    
    if (err != cudaSuccess) {
        AT_ERROR("CUDA chunk check kernel failed: ", cudaGetErrorString(err));
    }
    
    // Process active chunks with ray casting.
    auto output = torch::empty_like(voxel_grid);
    
    dim3 block = get_block_size();
    dim3 grid = get_grid_size(dim_x, dim_y, dim_z, block);
    
    fill_hollow_chunked_kernel<<<grid, block>>>(
        voxel_grid.data_ptr<uint8_t>(),
        output.data_ptr<uint8_t>(),
        priority_map.data_ptr<int>(),
        active_chunks.data_ptr<bool>(),
        dim_x,
        dim_y,
        dim_z,
        chunk_size_x,
        chunk_size_y,
        chunk_size_z,
        num_chunks_x,
        num_chunks_y,
        num_chunks_z,
        target_class,
        target_priority
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA fill hollow kernel failed: ", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
    
    return output;
}

torch::Tensor morphological_close_3d_cuda(
    torch::Tensor voxel_grid,
    int target_class,
    int kernel_size
)
{
    const int dim_x = voxel_grid.size(0);
    const int dim_y = voxel_grid.size(1);
    const int dim_z = voxel_grid.size(2);
    
    const int radius = kernel_size / 2;
    
    // Allocate intermediate and output tensors.
    auto dilated = torch::empty_like(voxel_grid);
    auto output = torch::empty_like(voxel_grid);
    
    dim3 block = get_block_size();
    dim3 grid = get_grid_size(dim_x, dim_y, dim_z, block);
    
    // Dilation.
    dilate_3d_kernel<<<grid, block>>>(
        voxel_grid.data_ptr<uint8_t>(),
        dilated.data_ptr<uint8_t>(),
        dim_x,
        dim_y,
        dim_z,
        target_class,
        radius
    );
    
    cudaError_t err = cudaGetLastError();
    
    if (err != cudaSuccess) {
        AT_ERROR("CUDA dilation kernel failed: ", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
    
    // Erosion (only for newly added voxels).
    erode_3d_kernel<<<grid, block>>>(
        voxel_grid.data_ptr<uint8_t>(),
        dilated.data_ptr<uint8_t>(),
        output.data_ptr<uint8_t>(),
        dim_x,
        dim_y,
        dim_z,
        target_class,
        radius
    );
    
    err = cudaGetLastError();
    
    if (err != cudaSuccess) {
        AT_ERROR("CUDA erosion kernel failed: ", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
    
    return output;
}
