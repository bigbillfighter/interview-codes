import torch
import torch.nn.functional as F

def roi_align(feature_map, rois, output_size):
    """
    实现ROI Align操作

    参数：
    - feature_map: 输入的特征图，大小为 [N, C, H, W]
    - rois: ROI区域的坐标，大小为 [num_rois, 4]，每行表示一个ROI的左上角和右下角坐标（x1, y1, x2, y2）
    - output_size: 输出的特征图大小，格式为 (output_height, output_width)

    返回值：
    - aligned_features: ROI Align操作后的特征，大小为 [num_rois, C, output_height, output_width]
    """

    # 获取ROI区域的数量
    num_rois = rois.size(0)

    # 获取特征图的通道数
    num_channels = feature_map.size(1)

    # 计算每个ROI的高度和宽度
    roi_heights = rois[:, 3] - rois[:, 1]
    roi_widths = rois[:, 2] - rois[:, 0]

    # 计算ROI区域在特征图上的每个单元格的大小
    bin_size_h = roi_heights / output_size[0]
    bin_size_w = roi_widths / output_size[1]

    # 初始化输出特征图
    aligned_features = torch.zeros(num_rois, num_channels, output_size[0], output_size[1], dtype=feature_map.dtype, device=feature_map.device)

    # 对每个ROI进行处理
    for i in range(num_rois):
        # 计算ROI的左上角坐标
        x1 = rois[i, 0]
        y1 = rois[i, 1]

        # 计算ROI的每个单元格的位置
        grid_h = torch.arange(output_size[0], dtype=torch.float32, device=feature_map.device) * bin_size_h[i] + y1
        grid_w = torch.arange(output_size[1], dtype=torch.float32, device=feature_map.device) * bin_size_w[i] + x1

        # 将每个单元格的位置转换为特征图上的坐标
        grid_h = torch.clamp(grid_h, 0, feature_map.size(2) - 1)
        grid_w = torch.clamp(grid_w, 0, feature_map.size(3) - 1)

        # 计算每个单元格的四个顶点的坐标
        grid_h_floor = torch.floor(grid_h).long()
        grid_w_floor = torch.floor(grid_w).long()
        grid_h_ceil = torch.ceil(grid_h).long()
        grid_w_ceil = torch.ceil(grid_w).long()

        # 计算每个单元格的四个顶点的权重
        h_weight = grid_h - grid_h_floor.float()
        w_weight = grid_w - grid_w_floor.float()

        # 在特征图上取出四个顶点的特征
        top_left_features = feature_map[:, :, grid_h_floor, grid_w_floor]
        top_right_features = feature_map[:, :, grid_h_floor, grid_w_ceil]
        bottom_left_features = feature_map[:, :, grid_h_ceil, grid_w_floor]
        bottom_right_features = feature_map[:, :, grid_h_ceil, grid_w_ceil]

        # 根据权重对四个顶点的特征进行插值
        top_left_weighted = (1 - h_weight) * (1 - w_weight) * top_left_features
        top_right_weighted = (1 - h_weight) * w_weight * top_right_features
        bottom_left_weighted = h_weight * (1 - w_weight) * bottom_left_features
        bottom_right_weighted = h_weight * w_weight * bottom_right_features

        # 将插值后的特征相加得到最终的插值结果
        interpolated_features = top_left_weighted + top_right_weighted + bottom_left_weighted + bottom_right_weighted

        # 将插值后的特征放入输出特征图中
        aligned_features[i] = interpolated_features

    return aligned_features