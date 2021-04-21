import numpy as np
import pickle
from graphviz import Digraph


class DecisionTree:

    def __init__(self, tree_type, is_continue, prune, max_depth):
        assert tree_type in ["ID3", "C4.5", "CART"], "Tree type unknown."
        self.tree_type = tree_type
        self.model = {"attribute": None, "attr_value": None, "num_samples": None, "metric_value": 0.0,
                      "class": None, "value": None, "child": [], "is_leaf": False}
        self.global_cnt = 0
        self.is_continue = is_continue
        self.max_depth = max_depth
        self.need_prune = prune
        self.label_range = None

    def metric(self, data, label, attr):
        """
        calculate metrics for certain tree type
        :param data: data subset
        :param label: label subset corresponding to data
        :param attr: attribute to be process -> int
        :return: the value of metric
        """
        raise NotImplementedError

    @staticmethod
    def __ent(data, label):
        """calculate entropy"""
        categories = np.unique(label)
        class_cnt = np.array(list(map(lambda x: len(data[label == x]), categories))).astype(np.int)
        p = class_cnt / class_cnt.sum()
        ent = -np.sum(p * np.log2(p))
        return ent

    def gini(self, label):
        """calculate gini index"""
        class_cnt = np.bincount(label, minlength=self.label_range)
        p = class_cnt / (class_cnt.sum() + 1e-6)
        gini_idx = 1 - np.sum(p * p)
        return gini_idx

    def get_info_gain(self, data, label, attr, need_pv=False):
        """calculate information gain"""
        entropy = self.__ent(data, label)
        attr_idxs = np.unique(data[:, attr])
        data_v = list(map(lambda x: data[data[:, attr] == x], attr_idxs))
        label_v = list(map(lambda x: label[data[:, attr] == x], attr_idxs))
        num_v = np.array([s.shape[0] for s in data_v]).astype(np.int)
        ent_v = np.array([self.__ent(x, y) for x, y in zip(data_v, label_v)])
        new_entropy = np.sum(num_v * ent_v / num_v.sum())
        if not need_pv:
            return entropy - new_entropy
        else:
            return (entropy - new_entropy), (num_v / num_v.sum())

    def get_gain_ratio(self, data, label, attr):
        """calculate gain ratio"""
        gain, p_v = self.get_info_gain(data, label, attr, need_pv=True)
        iv = -np.sum(p_v * np.log2(p_v))
        return gain / iv

    def get_gini_index(self, data, label, attr):
        """calculate the best split point given attr"""
        attr_idxs = np.unique(data[:, attr])
        splitter = None
        if not self.is_continue[attr]:
            # discrete
            # label_v = list(map(lambda x: label[data[:, attr] == x], attr_idxs))
            label_v = [label[data[:, attr] == x] for x in attr_idxs]
            # label_not_v = list(map(lambda x: label[data[:, attr] != x], attr_idxs))
            label_not_v = [label[data[:, attr] != x] for x in attr_idxs]
        else:
            # continue
            sorted_attr = np.sort(attr_idxs)
            splitter = np.convolve(sorted_attr, np.ones(2) / 2, 'valid')  # mean of 2 neighbor elements
            # label_v = list(map(lambda x: label[data[:, attr] <= x], splitter))
            # label_not_v = list(map(lambda x: label[data[:, attr] > x], splitter))
            label_v = [label[data[:, attr] <= x] for x in splitter]
            label_not_v = [label[data[:, attr] > x] for x in splitter]
        num_v = np.array([s.shape[0] for s in label_v])
        num_not_v = np.array([s.shape[0] for s in label_not_v])
        gini_v1 = np.array([self.gini(y) for y in label_v])
        gini_v2 = np.array([self.gini(y) for y in label_not_v])
        gini_index = (num_v * gini_v1 + num_not_v * gini_v2) / len(data)
        best_splitter_idx = np.argmin(gini_index)
        best_gini = gini_index[best_splitter_idx]
        best_splitter = attr_idxs[best_splitter_idx] if splitter is None else splitter[best_splitter_idx]
        return (best_gini, best_splitter)

    def fit(self, dataset, labels, attr_set=None):
        """build the tree model given input"""
        # self.__metric(dataset, labels, 0)
        self.label_range = len(np.unique(labels))
        if self.is_continue is None:
            self.is_continue = np.array([0] * dataset.shape[1])
        if attr_set is None:
            attr_set = np.arange(dataset.shape[1])
        self.tree_generate(dataset, labels, attr_set, model=self.model)

    def tree_generate(self, dataset, labels, attr_set, model, depth=0):
        categories = np.unique(labels)
        if len(categories) == 1:
            model["class"] = categories[0]
            model["is_leaf"] = True
            model["num_samples"] = len(dataset)
            return
        # if A is empty or D[:,A] are all the same
        if not len(attr_set) or (dataset[:, None, attr_set] == dataset[:, attr_set]).all():
            class_id = np.argmax(list(map(lambda x: len(dataset[labels == x]), categories)))
            model["class"] = categories[class_id]
            model["is_leaf"] = True
            model["num_samples"] = len(dataset)
            return
        # choose the best feature value a*
        metric_list = [self.metric(dataset, labels, attr) for attr in attr_set]
        best_attr_idx = int(np.argmax(metric_list))
        best_attr = attr_set[best_attr_idx]
        model["attribute"] = best_attr
        model["num_samples"] = len(dataset)
        model["metric_value"] = metric_list[best_attr_idx]
        # split nodes
        best_attr_values = np.unique(dataset[:, best_attr])
        for value in best_attr_values:
            sub_dict = {"attribute": best_attr, "attr_value": value, "num_samples": None, "metric_value": 0.0,
                        "class": None, "child": [], "is_leaf": False}
            model["child"].append(sub_dict)
            sub_dataset = dataset[dataset[:, best_attr] == value]
            sub_labels = labels[dataset[:, best_attr] == value]
            if not len(sub_labels):
                sub_dict["is_leaf"] = True
                sub_dict["class"] = categories[np.argmax(list(map(lambda x: len(dataset[labels == x]), categories)))]
                sub_dict["num_samples"] = len(dataset)
                # return
            else:
                attr_subset = np.delete(attr_set, best_attr_idx)
                self.tree_generate(sub_dataset, sub_labels, attr_subset, sub_dict)

    def predict(self, data):
        ...

    def prune(self, trainset, valset, valLabel):
        ...

    def evaluate(self, predicts, ground_truth):
        """ evaluate model """
        assert len(predicts) == len(ground_truth), "predicts and ground truth must have the same dim!"
        acc = (predicts == ground_truth).sum() / predicts.shape[0]
        return acc

    def save_model(self, model_path):
        with open(model_path, "wb") as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model saved!")

    def load_model(self, model_path):
        with open(model_path, "rb") as f:
            self.__dict__ = pickle.load(f)
        assert self.model is not None
        print("Model load finished")

    def visualize(self, attr_names, save_path):
        """visualize tree model"""
        g = Digraph("Decision Tree", filename=save_path)
        g.attr("node", shape="box")
        self.draw_graph(g, None, self.model, None, attr_names)
        g.view()

    def draw_graph(self, graph, parent_name, model, parent_attr, attr_names):
        fmt = f"attribute: {model['attribute']}\n" \
              f"num_samples: {model['num_samples']}\n" \
              f"metric_value: {round(model['metric_value'], 2)}\n" \
            if model['class'] is None else f"class: {attr_names[-1][model['class']]}"

        cur_name = f"node_{self.global_cnt}"
        self.global_cnt += 1
        graph.node(cur_name, label=fmt)
        if parent_name is not None:
            graph.edge(parent_name, cur_name, label=f"{attr_names[parent_attr][model['attr_value']]}")
        if not len(model['child']):
            return
        else:
            for node in model['child']:
                self.draw_graph(graph, cur_name, node, model['attribute'], attr_names)
