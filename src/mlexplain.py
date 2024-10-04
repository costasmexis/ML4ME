import pandas as pd
import numpy as np
import shap

def get_shap_values(model, X: pd.DataFrame, explainer_type: str = 'tree') -> pd.DataFrame:
    if explainer_type == 'tree':
        explainer = shap.TreeExplainer(model)
    else:
        raise ValueError("Only 'tree' explainer types are supported.")
    shap_values = explainer.shap_values(X)
    return pd.DataFrame(shap_values, columns=X.columns)