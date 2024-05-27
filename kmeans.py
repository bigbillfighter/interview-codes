import matplotlib.pyplot as plt
import random

import torch

class KMeans:
    def __init__(self, points, n_clusters=2, max_iter=10, decrese_threshold=1e-5):
        assert n_clusters is not None
        assert max_iter is not None

        self.points = points
        self.num_points = points.shape[0]
        self.n_clusters = n_clusters

        init_row = torch.randint(0, self.num_points, (self.n_clusters,))
        init_points = self.points[init_row]
        self.centers = init_points

        self.labels = None
        self.dists = None  
        self.mean_dists = torch.Tensor([float("Inf")])

        self.decrese_threshold = decrese_threshold
        self.max_iter = max_iter

    def fit_predict(self):
        for _ in range(self.max_iter):
            need_update = self.nearest_center()
            if need_update:
                for i in range(self.n_clusters):
                    self.centers[i] = self.points[self.labels == i].mean(dim=0)
            else:
                break


    def nearest_center(self):
        dists = (self.points[:, None, :] - self.centers[None, :, :]) ** 2
        dists = dists.sum(dim=-1)
        labels = dists.argmin(dim=-1)

        self.labels = labels
        self.dists = dists

        if(self.mean_dists - dists.mean() > self.decrese_threshold):
            self.mean_dists = dists.mean()
            return True
        
        return False

    @property
    def representative_samples(self):
        # 查找距离中心点最近的样本，作为聚类的代表样本，更加直观
        return torch.argmin(self.dists, dim=0)

if __name__ == "__main__":

    x1 = torch.randn((100, 2))
    mean1 = torch.tensor([2, 2])[None, :]
    std1 = torch.tensor([1, 1])[None, :]
    x1 = (x1 + mean1) * std1

    x2 = torch.randn((100, 2))
    mean2 = torch.tensor([-2, -2])[None, :]
    std2 = torch.tensor([1, 1])[None, :]
    x2 = (x2 + mean2) * std2

    X = torch.cat([x1, x2], dim=0)
    indices = list(range(200))
    random.shuffle(indices)
    X = X[indices]

    model = KMeans(points=X, n_clusters=2, max_iter=10)
    model.fit_predict()
    plt.scatter(model.points[:, 0], model.points[:, 1], c=model.labels, s=100, cmap='RdYlGn')
    plt.scatter(model.centers[:, 0], model.centers[:, 1], c='yellow', s=300, alpha=.8)
    plt.scatter(X[model.representative_samples][:, 0], X[model.representative_samples][:, 1], c='blue', s=300, alpha=.8)

    plt.show()



