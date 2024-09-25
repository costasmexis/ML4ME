import re

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree, export_text, plot_tree

SEED = 42

def split_string(s):
    return re.split(" >=| <=| >| <", s)


class MyClass:
    def __init__(self, df: pd.DataFrame, target: str, tree_clf: DecisionTreeClassifier):
        self.tree_clf = tree_clf
        self.target = target

        self.feature_names = df.drop(target, axis=1).columns.values.tolist()
        self.class_names = df[target].unique().tolist()

        self.rules = None
        self.cleaned_rules = None

    def get_rules(self):
        """
        Generate a list of rules from a decision tree.

        Args:
            tree (sklearn.tree.DecisionTreeClassifier): The decision tree model.
            feature_names (list): List of feature names.
            class_names (list): List of class names.

        Returns:
            list: List of rules generated from the decision tree.
        """
        tree_ = self.tree_clf.tree_
        feature_name = [
            self.feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        paths = []
        path = []

        def recurse(node, path, paths):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                p1, p2 = list(path), list(path)
                p1 += [f"({name} <= {np.round(threshold, 3)})"]
                recurse(tree_.children_left[node], p1, paths)
                p2 += [f"({name} > {np.round(threshold, 3)})"]
                recurse(tree_.children_right[node], p2, paths)
            else:
                path += [(tree_.value[node], tree_.n_node_samples[node])]
                paths += [path]

        recurse(0, path, paths)

        # sort by samples count
        samples_count = [p[-1][1] for p in paths]
        ii = list(np.argsort(samples_count))
        paths = [paths[i] for i in reversed(ii)]

        self.rules = []
        for path in paths:
            rule = "if "

            for p in path[:-1]:
                if rule != "if ":
                    rule += " and "
                rule += str(p)
            rule += " then "
            if self.class_names is None:
                rule += "response: " + str(np.round(path[-1][0][0][0], 3))
            else:
                classes = path[-1][0][0]
                l = np.argmax(classes)
                rule += f"class: {np.abs(1-self.class_names[1-l>0])} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
            rule += f" | based on {path[-1][1]:,} samples"
            self.rules += [rule]
    
    # TODO:: Implement this method
    def get_rules_df(self):
        rules = export_text(self.tree_clf, feature_names=self.tree_clf.feature_names_in_)
        # Get th3e 
        pass
    