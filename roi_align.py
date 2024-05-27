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



    

    

