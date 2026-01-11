// Academic Software License: Copyright © 2026 Goodarz Mehr.

// C++ bindings for CUDA voxel filling extension.

#include <torch/extension.h>

// Forward declarations of CUDA functions.
torch::Tensor fill_hollow_voxels_cuda(
    torch::Tensor voxel_grid,
    torch::Tensor priority_map,
    int target_class,
    int chunk_size_x,
    int chunk_size_y,
    int chunk_size_z
);

torch::Tensor morphological_close_3d_cuda(torch::Tensor voxel_grid, int target_class, int kernel_size);

// C++ interface for voxel filling using ray casting.
torch::Tensor fill_hollow_voxels(
    torch::Tensor voxel_grid,
    torch::Tensor priority_map,
    int target_class,
    int chunk_size_x = 200,
    int chunk_size_y = 200,
    int chunk_size_z = 80)
{
    // Validate inputs.
    TORCH_CHECK(voxel_grid.is_cuda(), "voxel_grid must be a CUDA tensor.");
    TORCH_CHECK(priority_map.is_cuda(), "priority_map must be a CUDA tensor.");
    TORCH_CHECK(voxel_grid.dim() == 3, "voxel_grid must be three-dimensional.");
    TORCH_CHECK(priority_map.dim() == 1 && priority_map.size(0) == 32, "priority_map must have shape (32,).");
    TORCH_CHECK(voxel_grid.is_contiguous(), "voxel_grid must be contiguous.");
    TORCH_CHECK(priority_map.is_contiguous(), "priority_map must be contiguous.");
    TORCH_CHECK(voxel_grid.dtype() == torch::kUInt8, "voxel_grid must be uint8.");
    TORCH_CHECK(priority_map.dtype() == torch::kInt32, "priority_map must be int32.");
    TORCH_CHECK(target_class >= 0 && target_class < 32, "target_class must be in [0, 31].");
    TORCH_CHECK(chunk_size_x > 0 && chunk_size_y > 0 && chunk_size_z > 0, "chunk sizes must be positive.");
    
    return fill_hollow_voxels_cuda(
        voxel_grid,
        priority_map,
        target_class,
        chunk_size_x,
        chunk_size_y,
        chunk_size_z
    );
}

// C++ interface for 3D morphological closing.
torch::Tensor morphological_close_3d(torch::Tensor voxel_grid, int target_class, int kernel_size)
{
    // Validate inputs.
    TORCH_CHECK(voxel_grid.is_cuda(), "voxel_grid must be a CUDA tensor.");
    TORCH_CHECK(voxel_grid.dim() == 3, "voxel_grid must be three-dimensional.");
    TORCH_CHECK(voxel_grid.is_contiguous(), "voxel_grid must be contiguous.");
    TORCH_CHECK(voxel_grid.dtype() == torch::kUInt8, "voxel_grid must be uint8.");
    TORCH_CHECK(target_class >= 0 && target_class < 32, "target_class must be in [0, 31].");
    TORCH_CHECK(kernel_size > 0 && kernel_size % 2 == 1, "kernel_size must be positive and odd.");
    
    return morphological_close_3d_cuda(voxel_grid, target_class, kernel_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fill_hollow_voxels_cuda", &fill_hollow_voxels,
        "Fill interiors of objects in a voxel grid using 6-direction ray casting (CUDA).",
        py::arg("voxel_grid"),
        py::arg("priority_map"),
        py::arg("target_class"),
        py::arg("chunk_size_x") = 200,
        py::arg("chunk_size_y") = 200,
        py::arg("chunk_size_z") = 80
    );
    m.def(
        "morphological_close_3d_cuda",
        &morphological_close_3d,
        "Apply 3D morphological closing.",
        py::arg("voxel_grid"),
        py::arg("target_class"),
        py::arg("kernel_size")
    );
}
