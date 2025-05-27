import numpy as np
from collections import Counter

def entropy(y):
    total = len(y)
    counts = Counter(y)
    return -sum((c / total) * np.log2(c / total) for c in counts.values())

def information_gain(y, x_column):
    values = set(x_column)
    total = len(y)
    weighted_entropy = 0
    for v in values:
        subset_y = [label for i, label in enumerate(y) if x_column[i] == v]
        weighted_entropy += (len(subset_y) / total) * entropy(subset_y)
    return entropy(y) - weighted_entropy

def majority_class(y):
    return Counter(y).most_common(1)[0][0]

class Node:
    def __init__(self, feature=None, branches=None, leaf_label=None):
        self.feature = feature
        self.branches = branches
        self.leaf_label = leaf_label

    def is_leaf(self):
        return self.leaf_label is not None

    def predict(self, x, feature_names):
        if self.is_leaf():
            return self.leaf_label
        feature_value = x[feature_names.index(self.feature)]
        if feature_value in self.branches:
            return self.branches[feature_value].predict(x, feature_names)
        else:
            return majority_class([branch.leaf_label for branch in self.branches.values() if branch.is_leaf()])

class DecisionTree:
    def __init__(self, X, y, threshold=1.0, max_depth=None):
        self.feature_names = ['name', 'color', 'format']
        self.tree = self.build_tree(X, y, self.feature_names, threshold, max_depth)

    def build_tree(self, X, y, feature_names, threshold, max_depth, depth=0):
        if not y or len(set(y)) == 1 or (max_depth is not None and depth >= max_depth):
            return Node(leaf_label=majority_class(y))

        best_gain = 0
        best_feature = None
        for i, feature in enumerate(feature_names):
            x_column = [row[i] for row in X]
            gain = information_gain(y, x_column)
            print(f'Feature: {feature}, Info Gain: {gain:.4f}')  # DEBUG

            if gain > best_gain:
                best_gain = gain
                best_feature = feature

        if best_gain < threshold or best_feature is None:
            return Node(leaf_label=majority_class(y))

        best_index = feature_names.index(best_feature)
        branches = {}
        values = set(row[best_index] for row in X)
        for val in values:
            sub_X = [row for row in X if row[best_index] == val]
            sub_y = [y[i] for i, row in enumerate(X) if row[best_index] == val]
            new_features = feature_names[:best_index] + feature_names[best_index+1:]
            sub_X = [[v for j, v in enumerate(row) if j != best_index] for row in sub_X]
            branches[val] = self.build_tree(sub_X, sub_y, new_features, threshold, max_depth, depth + 1)

        return Node(feature=best_feature, branches=branches)

    def predict(self, x):
        return self.tree.predict(x, self.feature_names)

def train_decision_tree(X, y):
    return DecisionTree(X, y, threshold=0.01, max_depth=5)
