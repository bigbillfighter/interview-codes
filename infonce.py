import torch
import torch.nn.functional as F

def info_nce_loss(query, positive_key, negative_keys, temperature=0.07):
    """
    计算 InfoNCE 损失。

    参数：
    - query: 查询向量，大小为 [batch_size, embedding_dim]
    - positive_key: 正样本向量，大小为 [batch_size, embedding_dim]
    - negative_keys: 负样本向量，大小为 [batch_size, num_negatives, embedding_dim]
    - temperature: 温度参数，用于缩放相似度

    返回值：
    - loss: InfoNCE 损失
    """

    batch_size = query.size(0)
    embedding_dim = query.size(1)
    num_negatives = negative_keys.size(1)

    # 计算查询向量与正样本向量的相似度
    positive_logit = torch.sum(query * positive_key, dim=-1, keepdim=True) / temperature

    # 计算查询向量与负样本向量的相似度
    negative_logits = torch.bmm(negative_keys, query.unsqueeze(2)).squeeze(-1) / temperature

    # 将正样本和负样本的相似度拼接
    logits = torch.cat((positive_logit, negative_logits), dim=1)

    # 创建标签，正样本标签为0
    labels = torch.zeros(batch_size, dtype=torch.long, device=query.device)

    # 计算 InfoNCE 损失
    loss = F.cross_entropy(logits, labels)

    return loss

# 示例用法
batch_size = 8
embedding_dim = 128
num_negatives = 16

# 随机生成查询向量、正样本向量和负样本向量
query = torch.randn(batch_size, embedding_dim)
positive_key = torch.randn(batch_size, embedding_dim)
negative_keys = torch.randn(batch_size, num_negatives, embedding_dim)

# 计算 InfoNCE 损失
loss = info_nce_loss(query, positive_key, negative_keys)
print("InfoNCE Loss:", loss.item())