import torch

def cal_area(bbox):
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def iou(bbox1: torch.Tensor, bbox2: torch.Tensor, eps=1e-8):
    """
    bbox1: (K1, 4)
    bbox2: (K2, 4)
    box: [x1, y1, x2, y2]

    return:
    iou: (K1, K2)
    """
    area1 = cal_area(bbox1)
    area2 = cal_area(bbox2)

    bbox1 = bbox1[:, None, :]
    bbox2 = bbox2[None, :, :]

    lt = torch.max(bbox1[:, :, :2], bbox2[:, :, :2])
    rb = torch.min(bbox1[:, :, 2:], bbox2[:, :, 2:])

    wh = (rb - lt).clamp(min=0) 
    inter = wh[:, :, 0] * wh[:, :, 1]
 
    _iou = inter / (area1[:, None] + area2[None, :] - inter)
    return _iou 


if __name__ == '__main__':

    bbox1 = torch.tensor([
        [0, 20, 100, 80],
        [40, 20, 100, 120],
        [200, 20, 220, 60],
    ], dtype=torch.float32)

    bbox2 = torch.tensor([
        [60, 20, 110, 200],
        [0, 90, 190, 100],
        [22, 38, 38, 48],
        [60, 0, 170, 200],
        [40, 90, 110, 180],

    ], dtype=torch.float32)

    print(iou(bbox1, bbox2))