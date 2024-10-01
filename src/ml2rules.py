import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree, export_text, plot_tree

from .machinelearning import train_decisiontree

def ml2tree(X_train: pd.DataFrame, y_train: pd.Series, scoring: str = 'accuracy', n_trials: int = 200) -> DecisionTreeClassifier:
    """
    Train a decision tree model from a dataset.

    Args:
        X_train (pd.DataFrame): The features of the dataset.
        y_train (pd.Series): The target of the dataset.

    Returns:
        DecisionTreeClassifier: The trained decision tree model.
    """
    tree_clf = train_decisiontree(X_train, y_train, scoring=scoring, n_trials=n_trials)
    return tree_clf

def sample_from_df(df: pd.DataFrame, rule: str, skope_rules: bool = False) -> pd.DataFrame:
        """
        Sample from a dataframe based on a rule.

        Args:
            df (pd.DataFrame): The dataframe.
            rule (str): The rule. Should be in the format of def get_rule_constraints (i.e., )
            skope_rules (bool): Whether the rule is generated from SkopeRules.
            
        Returns:
            pd.DataFrame: The sampled dataframe.
        """
        if skope_rules:
            rule = rule.split('and')
            rule = [r.replace(' ', '') for r in rule]
            
        for i in rule:
            if '<=' in i:
                parameter = i.split('<=')[0]
                value = i.split('<=')[1]
                df = df[df[parameter] <= float(value)]
            elif '>' in i:
                parameter = i.split('>')[0]
                value = i.split('>')[1] 
                df = df[df[parameter] > float(value)]
            else:
                raise ValueError('No operator found')
        return df
    
class TreeRuler:
    def __init__(self, df: pd.DataFrame, target: str, tree_clf: DecisionTreeClassifier):
        self.tree_clf = tree_clf
        self.target = target

        self.feature_names = df.drop(target, axis=1).columns.values.tolist()
        self.class_names = df[target].unique().tolist()

        self.rules = None
        self.cleaned_rules = None
        
    def visualize_tree(self, save_path: str = None) -> None:
        """
        Visualize the decision tree.

        Args:
            tree (sklearn.tree.DecisionTreeClassifier): The decision tree model.
            feature_names (list): List of feature names.
            class_names (list): List of class names.
            save_path (str): Path to save the tree plot.
        """
        plot_tree(self.tree_clf, filled=True, feature_names=self.feature_names)
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def get_rules(self) -> None:
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
                rule += f"class: {self.class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
                # rule += f"class: {np.abs(self.class_names[1-l>0])} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
            rule += f" | based on {path[-1][1]:,} samples"
            self.rules += [rule]
                
    
    def get_rule_constraints(self, rule_index: int) -> str:
        """
        Get the constraints from a rule.

        Args:
            rule (str): The rule.

        Returns:
            str: The constraints of the rule.
        """
        rule = self.rules[rule_index]
        rule = rule.replace("if ", "")
        rule = rule.replace('and', '')
        rule = rule.split('then')[0].strip()
        rule = rule.split(') ')
        rule = [i.replace('(', '') for i in rule]
        rule = [i.replace(')', '') for i in rule]
        rule = [i.replace(' ', '') for i in rule]
        return rule
    
    