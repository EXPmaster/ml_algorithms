import numpy as np


class NaiveBayes:
    """ 所有属性视为高斯分布连续属性的朴素贝叶斯分类器 """
    def __init__(self, alpha=1e-9):
        self.alpha = alpha  # 方差平滑因子
        self.mu = None
        self.var = None
        self.label_prob = None

    def fit(self, dataset, labels):
        """训练模型，计算每个属性的均值和方差"""
        label_cnt = np.bincount(labels)
        self.label_prob = np.log(label_cnt / len(labels))
        # attr_cnt = [[np.bincount((dataset[labels == label_idx])[:, i]) for i in range(dataset.shape[1])]
        #             for label_idx in range(len(label_cnt))]
        self.mu = [np.mean(dataset[labels == i], axis=0) for i in range(len(label_cnt))]
        self.var = [np.var(dataset[labels == i], axis=0) for i in range(len(label_cnt))]
        epsilon = np.max(self.var) * self.alpha  # 平滑系数
        self.var += epsilon

    @staticmethod
    def __pdf(x, mu, var):
        """
        计算概率密度，满足高斯分布
        这里对它取对数避免连乘，减小计算量
        """
        pdf = -0.5 * np.log(2 * np.pi * var) - (x - mu) ** 2 / (2 * var)
        return pdf

    def predict(self, dataset):
        return np.array([self.__apply_batch_predict(data) for data in dataset])

    def __apply_batch_predict(self, data):
        hypothesis = [self.label_prob[i] + np.sum(self.__pdf(data, self.mu[i], self.var[i]))
                      for i in range(len(self.label_prob))]
        return np.argmax(hypothesis)

    def evaluate(self, preds, labels, metric='err'):
        assert len(preds) == len(labels)
        if metric == 'err':
            return (preds != labels).sum() / len(labels)
        elif metric == 'acc':
            return (preds == labels).sum() / len(labels)
        else:
            raise TypeError('Unknown metric type!')
