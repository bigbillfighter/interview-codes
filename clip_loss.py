import torch
import torch.nn.functional as F

def clip_loss(image_features, text_features, image_projection, text_projection, temperature=0.07):
    """
    计算 CLIP 的对比损失

    参数：
    - image_features: 图像的特征嵌入，大小为 [batch_size, embedding_dim]
    - text_features: 文本的特征嵌入，大小为 [batch_size, embedding_dim]
    - temperature: 温度参数，用于缩放相似度

    返回值：
    - loss: CLIP 的对比损失
    """
    # projection
    image_embeddings = image_projection(image_features)
    text_embeddings = text_projection(text_features)

    image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

    # 计算相似性矩阵
    logits_per_image = torch.matmul(image_features, text_features.T) / temperature
    logits_per_text = logits_per_image.t()

    # 创建标签
    batch_size = image_features.size(0)
    labels = torch.arange(batch_size, dtype=torch.long)

    # 计算图像和文本的交叉熵损失
    loss_image = F.cross_entropy(logits_per_image, labels)
    loss_text = F.cross_entropy(logits_per_text, labels)

    # 平均损失
    loss = (loss_image + loss_text) / 2.0
    return loss


if __name__ == "__main__":
    # 示例用法
    batch_size = 8
    embedding_dim = 512

    # 随机生成图像和文本的特征嵌入
    image_features = torch.randn(batch_size, embedding_dim)
    text_features = torch.randn(batch_size, embedding_dim)

    # 计算 CLIP 损失
    loss = clip_loss(image_features, text_features, image_projection=torch.nn.Identity(), text_projection=torch.nn.Identity())
    print("CLIP Loss:", loss.item())

