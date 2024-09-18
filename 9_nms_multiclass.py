import torch

def cal_area(bbox):
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])

def iou(bbox1: torch.Tensor, bbox2: torch.Tensor, eps=0):
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

    wh = (rb - lt).clamp(min=eps) 
    inter = wh[:, :, 0] * wh[:, :, 1]
 
    _iou = inter / (area1[:, None] + area2[None, :] - inter)
    return _iou  

def nms_multiclass(bboxes: torch.Tensor, labels:torch.Tensor, threshold=0.7):
    bboxes_keep = []
    labels_keep = []
    for cls in torch.unique(labels):
        orders = torch.where(cls == labels)[0]
        bboxes_single_class = bboxes[orders]
        scores = bboxes_single_class[:, -1]
        boxes = bboxes_single_class[:, 0:4]

        idxes = scores.argsort(dim=0, descending=True)
        while idxes.shape[0] > 0:
            max_score_idx = idxes[0]
            max_score_box = boxes[max_score_idx][None, :]
            bboxes_keep.append(bboxes_single_class[max_score_idx])
            labels_keep.append(torch.tensor([cls], dtype=torch.int64))

            idxes = idxes[1:]
            if idxes.shape[0] > 0:
                left_boxes = boxes[idxes]
                ious = iou(max_score_box, left_boxes)
                idxes = idxes[ious[0] < threshold]
        
    bboxes_keep = torch.stack(bboxes_keep, dim=0)
    labels_keep = torch.stack(labels_keep, dim=0)

    return bboxes_keep, labels_keep

if __name__ == "__main__":

    bboxes = torch.tensor([
        [100, 100, 210, 210, 0.72],
        [250, 250, 420, 420, 0.8],
        [220, 220, 320, 330, 0.92],
        [100, 100, 210, 210, 0.72],
        [230, 240, 325, 330, 0.81],
        [220, 230, 315, 340, 0.9]
    ])

    labels = torch.tensor([
        [1],
        [1],
        [0],
        [2],
        [0],
        [0]
    ], dtype=torch.int64)

    bboxes_keep, labels_keep = nms_multiclass(bboxes, labels)
    print(bboxes_keep)
    print(labels_keep)
