import numpy as np
from BasicTree import DecisionTree
from copy import deepcopy
from joblib import Parallel, delayed


class ID3Tree(DecisionTree):

    def __init__(self, is_continue=None, prune=False, max_depth=10):
        tree_type = "ID3"
        super().__init__(tree_type, is_continue, prune, max_depth)

    def metric(self, data, label, attr):
        return self.get_info_gain(data, label, attr)


class C45Tree(DecisionTree):

    def __init__(self, is_continue=None, prune=False, max_depth=10):
        tree_type = "C4.5"
        super().__init__(tree_type, is_continue, prune, max_depth)

    def metric(self, data, label, attr):
        return self.get_gain_ratio(data, label, attr)


class CartTree(DecisionTree):

    def __init__(self, is_continue=None, prune=False, max_depth=10, gini_thresh=0.01, sample_thresh=1):
        tree_type = "CART"
        super().__init__(tree_type, is_continue, prune, max_depth)
        self.gini_thres = gini_thresh
        self.sample_thres = sample_thresh

    def metric(self, data, label, attr):
        return self.get_gini_index(data, label, attr)

    def tree_generate(self, dataset, labels, attr_set, model, depth=0):
        # if A is empty or D[:,A] are all the same or reaches the max depth
        # if not len(attr_set) or (dataset[:, None, attr_set] == dataset[:, attr_set]).all() or depth > self.max_depth:
        if (not len(attr_set)) or len(dataset) <= self.sample_thres or\
                self.gini(labels) < self.gini_thres or depth >= self.max_depth:
            class_values = np.bincount(labels, minlength=self.label_range)
            class_id = np.argmax(class_values)
            model["value"] = class_values
            model["class"] = class_id
            model["is_leaf"] = True
            model["num_samples"] = len(dataset)
            return
        # choose the best feature value a*

        # metric_splitter_list = [self.metric(dataset, labels, attr) for attr in attr_set]
        metric_splitter_list = Parallel(n_jobs=12)(delayed(self.metric)(dataset, labels, attr) for attr in attr_set)
        metric_list = [x[0] for x in metric_splitter_list]
        splitter_list = [x[1] for x in metric_splitter_list]
        best_attr_idx = int(np.argmin(metric_list))
        best_attr = attr_set[best_attr_idx]
        best_splitter = splitter_list[best_attr_idx]
        model["attribute"] = best_attr
        model["num_samples"] = len(dataset)
        model["metric_value"] = metric_list[best_attr_idx]
        model["attr_value"] = best_splitter

        # split nodes
        new_depth = depth + 1
        print(new_depth)
        split_idx = dataset[:, best_attr] <= best_splitter if self.is_continue[best_attr] else \
            dataset[:, best_attr] == best_splitter
        attr_subset = np.delete(attr_set, best_attr_idx)
        for i in range(2):
            sub_dict = {"attribute": best_attr, "attr_value": None, "num_samples": None, "metric_value": 0.0,
                        "class": None, "value": None, "child": [], "is_leaf": False}
            model["child"].append(sub_dict)
            sub_dataset = dataset[split_idx]
            sub_labels = labels[split_idx]
            split_idx = ~split_idx

            if not len(sub_labels):
                class_values = np.bincount(labels, minlength=self.label_range)
                class_id = np.argmax(class_values)
                sub_dict["is_leaf"] = True
                sub_dict["class"] = class_id
                sub_dict["value"] = class_values
                sub_dict["num_samples"] = len(dataset)
                # return
            else:
                self.tree_generate(sub_dataset, sub_labels, attr_subset, sub_dict, new_depth)

    def prune(self, trainset, valset, valLabel):
        print("pruning...")
        trees = [self.model]
        model = self.model
        while not (model["child"][0]["is_leaf"] and model["child"][1]["is_leaf"]):
            model = self.__apply_prune(trainset, model)
            trees.append(model)
        print(len(trees))
        acc = []
        for i, m in enumerate(trees):
            self.model = m
            preds = self.predict(valset)
            acc.append(self.evaluate(preds, valLabel))
            # self.visualize([np.array([1, 2, 3, 4])], save_path=f"Task2/tmp/tree_res{i}.gv")
        print(acc)
        idx = int(np.argmax(acc))
        best_acc = acc[idx]
        print(f"acc after prune: {best_acc}")
        self.model = trees[idx]
        self.visualize([np.array([1, 2, 3, 4])], save_path=f"Task3/tmp/tree_prune.gv")

    def __apply_prune(self, trainset, model):
        model = deepcopy(model)
        stack = [model]
        nodes = []
        while len(stack):
            cur_model = stack.pop()
            assert len(cur_model["child"]) == 2
            if not cur_model["child"][0]["is_leaf"]:
                stack.append(cur_model["child"][0])
            if not cur_model["child"][1]["is_leaf"]:
                stack.append(cur_model["child"][1])
            nodes.append(cur_model)

        nodes.reverse()
        for node in nodes:
            node["leaf_cnt"] = sum(list(map(lambda x: int(x["is_leaf"]) if x["is_leaf"] else x["leaf_cnt"], node["child"])))
            node["value"] = node["child"][0]["value"] + node["child"][1]["value"]
            node["class"] = np.argmax(node["value"])
            c_t = self.gini(self.predict(trainset, node))
            tmp_node = deepcopy(node)
            tmp_node["child"][0]["is_leaf"] = True
            tmp_node["child"][1]["is_leaf"] = True
            c_tt = self.gini(self.predict(trainset, tmp_node))
            node["g_t"] = (c_t - c_tt) / (node["leaf_cnt"] - 1)

        best_cut_node = nodes[int(np.argmin(list(map(lambda x: x["g_t"], nodes))))]
        best_cut_node["is_leaf"] = True
        # print(len(values))
        # print([(v["attribute"], v["leaf_cnt"]) for v in nodes])
        return model

    def predict(self, batch_data, in_model=None):
        tmp_model = self.model
        self.model = in_model if in_model is not None else self.model
        if in_model is not None:
            results = np.array(list(map(self.__apply_batch_prediction, batch_data)))
        else:
            results = np.array(list(map(self.__apply_batch_prediction, batch_data)))
        self.model = tmp_model
        return results

    def __apply_batch_prediction(self, data):
        model = self.model
        while not model["is_leaf"]:
            assert(len(model["child"]) == 2)
            if data[model["attribute"]] <= model["attr_value"]:
                model = model["child"][0]
            else:
                model = model["child"][1]

        return model["class"]

    def draw_graph(self, graph, parent_name, model, parent_attr, attr_names):
        if self.is_continue[model['attribute']]:
            fmt = f"attribute {model['attribute']} <= {round(model['attr_value'], 3)}\n" \
                  f"num_samples: {model['num_samples']}\n" \
                  f"value: {model['value']}\n" \
                  f"metric_value: {round(model['metric_value'], 2)}\n" \
                if not model['is_leaf'] else f"value: {model['value']}\n" \
                                             f"class: {attr_names[-1][int(model['class'])]}"
        else:
            fmt = f"attribute {model['attribute']} == {round(model['attr_value'], 3)}\n" \
                  f"num_samples: {model['num_samples']}\n" \
                  f"value: {model['value']}\n" \
                  f"metric_value: {round(model['metric_value'], 2)}\n" \
                if not model['is_leaf'] else f"value: {model['value']}\n" \
                                             f"class: {attr_names[-1][int(model['class'])]}"

        cur_name = f"node_{self.global_cnt}"
        self.global_cnt += 1
        graph.node(cur_name, label=fmt)
        if parent_name is not None:
            graph.edge(parent_name, cur_name, label=f"{bool(parent_attr)}")
        if model['is_leaf']:
            return
        else:
            for i, node in enumerate(model['child']):
                self.draw_graph(graph, cur_name, node, 1-i, attr_names)
