import sys

sys.path.append('..')

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from mrmr import mrmr_classif

from src.machinelearning import (
    ANNClassifier,
    evaluate_ann,
    evaluate_sklearn,
    get_predictions,
    train_ann,
    train_decisiontree,
    train_skoperules,
    train_xgboost,
    cross_val_sklearn
)
from src.ml2rules import TreeRuler, ml2tree, sample_from_df
from src.utils import get_cc_mat, get_dataset, non_stratify_split, stratify_split


df = get_dataset(
    labels_file="../data/class_vector_train_ref.mat",
    params_file="../data/training_set_ref.mat",
    names_file="../data/paremeterNames.mat",
)

print(f"***Dataset shape: {df.shape}\n")

X_train, X_test, y_train, y_test = stratify_split(data=df, train_size=100000, target="label")

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Feature selection using MR-MR
K = int(100)
print(f"***Number of features to select: {K}")
selected_features = mrmr_classif(X=train_df.drop("label", axis=1), y=train_df["label"], K=K)

# keep only selected features
X_train = X_train[selected_features]
X_test = X_test[selected_features]

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Pickle save selected features
with open('../models/selected_features_100000.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

# Decision Tree Training
cart_model = train_decisiontree(X_train, y_train, scoring='matthews_corrcoef', n_trials=500)
with open('../models/cart_model_100000.pkl', 'wb') as f:
    pickle.dump(cart_model, f)
    

# Skope-Rules training
skope_rules_clf = train_skoperules(X_train, y_train, scoring='matthews_corrcoef', n_iter=500)
with open('../models/skope_rules_clf_100000.pkl', 'wb') as f:
    pickle.dump(skope_rules_clf, f)
    
# XGBoost training
xgb_model = train_xgboost(X_train, y_train, scoring='matthews_corrcoef', n_trials=500)
with open('../models/xgb_model_100000.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

xgb_preds = xgb_model.predict(X_train)
xgb2tree = ml2tree(X_train, xgb_preds, scoring='matthews_corrcoef', n_trials=500)
with open('../models/xgb2tree_100000.pkl', 'wb') as f:
    pickle.dump(xgb2tree, f)
    


    