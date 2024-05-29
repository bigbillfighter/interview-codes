import torch
import torch.nn.functional as F

def roi_align(feature_map, rois, output_size):
    """
    params:
        feature_map: [N, C, H, W]
        rois: [N, K, 4]
        output_size: (h', w')
    output:
        output_features: [N, K, C, h', w']
    """
    N, K = rois.shape[:2]
    C = feature_map.shape[1]

    roi_heights = rois[:, :, 3] - rois[:, :, 1]
    roi_widths = rois[:, :, 2] - rois[:, :, 0]

    bin_size_h = roi_heights / output_size[0]
    bin_size_w = roi_widths / output_size[1]

    output_features = torch.zeros(N, K, C, output_size[0], output_size[1], dtype=feature_map.dtype)

    for b in range(N):
        for k in range(K):
            # top left
            x1, y1 = rois[b, k, 0], rois[b, k, 1]
            grid_h = torch.arange(output_size[0], dtype=torch.float32) * bin_size_h[b, k] + y1 
            grid_w = torch.arange(output_size[1], dtype=torch.float32) * bin_size_w[b, k] + x1

            grid_h = torch.clamp(grid_h, 0, feature_map.size(2) - 1)
            grid_w = torch.clamp(grid_w, 0, feature_map.size(3) - 1)

            grid_h_expanded = grid_h[:, None].repeat(1, len(grid_w)).flatten()
            grid_w_expanded = grid_w[None, :].repeat(len(grid_h), 1).flatten()

            grid_h_floor = torch.floor(grid_h_expanded).to(torch.int64)
            grid_w_floor = torch.floor(grid_w_expanded).to(torch.int64)
            grid_h_ceil = torch.ceil(grid_h_expanded).to(torch.int64)
            grid_w_ceil = torch.ceil(grid_w_expanded).to(torch.int64)

            h_weight = grid_h_expanded - grid_h_floor.float()
            w_weight = grid_w_expanded - grid_w_floor.float()

            top_left_features = feature_map[b, :, grid_h_floor, grid_w_floor]
            top_right_features = feature_map[b, :, grid_h_floor, grid_w_ceil]
            bottom_left_features = feature_map[b, :, grid_h_ceil, grid_w_floor]
            bottom_right_features = feature_map[b, :, grid_h_ceil, grid_w_ceil]

            top_left_features = (1 - h_weight) * (1 - w_weight) * top_left_features
            top_right_features = (1 - h_weight) * w_weight * top_right_features
            bottom_left_features = h_weight * (1 - w_weight) * bottom_left_features
            bottom_right_features = h_weight * h_weight * bottom_right_features

            interpolated_features = top_left_features + top_right_features + bottom_left_features + bottom_right_features

            output_features[b, k] = interpolated_features.reshape(C, *output_size)

    return output_features


if __name__ == "__main__":
    features = torch.randn(2, 8, 10, 10)
    rois = torch.tensor([[
        [0.5, 1.0, 3.2, 4.4],
        [3.5, 2.0, 4.7, 9.4],
        [1.3, 2.9, 8.6, 8.8],
    ], [
        [1.5, 4.2, 9.2, 6.4],
        [3.0, 0.0, 8.7, 6.4],
        [3.3, 6.9, 4.4, 9.8],
    ]], dtype=torch.float32)

    output_size = (3, 3)
    aligned_features = roi_align(features, rois, output_size)

    print(aligned_features)



    

    

