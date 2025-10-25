// Academic Software License: Copyright © 2025 Goodarz Mehr.

// C++ bindings for CUDA bbox extension.

#include <torch/extension.h>

// Forward declaration of CUDA function.
torch::Tensor num_inside_bbox_cuda(torch::Tensor points, torch::Tensor bbox);

// C++ interface.
torch::Tensor num_inside_bbox(torch::Tensor points, torch::Tensor bbox) {
    // Validate inputs.
    TORCH_CHECK(points.is_cuda(), "points must be a CUDA tensor.");
    TORCH_CHECK(bbox.is_cuda(), "bbox must be a CUDA tensor.");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "points must have shape (N, 3).");
    TORCH_CHECK(bbox.dim() == 1 && bbox.size(0) == 24, "bbox must be flattened (24,).");
    TORCH_CHECK(points.is_contiguous(), "points must be contiguous.");
    TORCH_CHECK(bbox.is_contiguous(), "bbox must be contiguous.");
    
    return num_inside_bbox_cuda(points, bbox);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("num_inside_bbox_cuda", &num_inside_bbox, "Check if points are inside a 3D bounding box (CUDA).");
}
