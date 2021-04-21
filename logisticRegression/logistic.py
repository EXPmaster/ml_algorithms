import numpy as np


class LogisticRegression:
    """
    Logistic Regression

    :param norm: 正则化值，为0则不进行正则化
    :param tol: 损失容限，小于它则停止迭代
    :param max_iter: 迭代次数上限
    :param lr: 学习率
    :param thresh: 正负类判决门限
    :param verbose: 是否打印loss值
    """

    def __init__(self, norm=0, tol=1e-4, max_iter=100, lr=1e-2, thresh=0.5, verbose=False):
        self.norm = norm
        self.tol = tol
        self.max_iter = max_iter
        self.lr = lr
        self.weights = None
        self.thresh = thresh
        self.verbose = verbose

    def fit(self, trainset, labels):
        """训练模型"""
        # 给训练集添加偏置项1
        train_bias = np.ones((trainset.shape[0], 1))
        trainset = np.concatenate([trainset, train_bias], axis=1)
        # 将y转置
        if labels.shape[-1] != 1:
            labels = labels.reshape(-1, 1)
        assert labels.shape == (len(trainset), 1)

        # 初始化权重，满足高斯分布
        self.weights = np.random.normal(size=(trainset.shape[1], 1))
        # 梯度下降法迭代
        for epoch in range(self.max_iter):
            loss = self.__get_loss(trainset, labels)
            if self.verbose:
                print(loss)
            if loss < self.tol:
                break
            grad = self.__calcu_grad(trainset, labels)
            self.weights -= self.lr * grad

    def __get_loss(self, x, y):
        """计算损失，带l2正则化"""
        tmp_dot = x.dot(self.weights)
        return np.mean(-y * tmp_dot + np.log(1 + np.exp(tmp_dot))) + \
               self.norm / 2 * np.sum(self.weights * self.weights)

    def __calcu_grad(self, x, y):
        """计算梯度"""
        tmp_dot = np.exp(x.dot(self.weights))
        multiplier = y - tmp_dot / (1 + tmp_dot)
        return -x.transpose().dot(multiplier) + self.norm * self.weights

    def predict(self, dataset):
        """预测"""
        # 添加偏置
        if dataset.shape[1] != self.weights.shape[0]:
            bias = np.ones((dataset.shape[0], 1))
            dataset = np.concatenate([dataset, bias], axis=1)
        # 计算类别
        logits = self.sigmoid(dataset.dot(self.weights)).reshape(-1)
        logits[logits >= self.thresh] = 1
        logits[logits < self.thresh] = 0
        return logits.astype(np.int)

    @staticmethod
    def evaluate(preds, gts):
        """评估"""
        return (preds == gts).sum() / len(preds)

    def score(self, dataset, gts):
        """输入测试样本计算精度"""
        preds = self.predict(dataset)
        return self.evaluate(preds, gts)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
