// Academic Software License: Copyright © 2025 Goodarz Mehr.
// C++ bindings for CUDA bbox extension

#include <torch/extension.h>

// Forward declaration of CUDA function
torch::Tensor is_inside_bbox_cuda(torch::Tensor points, torch::Tensor bbox);

// C++ interface
torch::Tensor is_inside_bbox(torch::Tensor points, torch::Tensor bbox) {
    // Validate inputs
    TORCH_CHECK(points.is_cuda(), "points must be a CUDA tensor");
    TORCH_CHECK(bbox.is_cuda(), "bbox must be a CUDA tensor");
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, 
                "points must have shape (N, 3)");
    TORCH_CHECK(bbox.numel() == 24, "bbox must have 24 elements (8 corners * 3 coords)");
    
    return is_inside_bbox_cuda(points.contiguous(), bbox.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("is_inside_bbox_cuda", &is_inside_bbox, 
          "Check if points are inside a 3D bounding box (CUDA)");
}
