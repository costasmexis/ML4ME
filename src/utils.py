import pandas as pd
from scipy.io import loadmat

def get_labels_mat(filename) -> list:
    mat_data = loadmat(f"./data/{filename}")
    label = mat_data["class_vector_train"].reshape(-1,)
    label = [item for sublist in label for item in sublist]
    return label


def get_parameters_mat(filename) -> pd.DataFrame:
    mat_data = loadmat(f"./data/{filename}")
    parameters = mat_data["training_set"]
    return pd.DataFrame(parameters.T)


def get_parameter_names_mat(filename) -> list:
    mat_data = loadmat("./data/paremeterNames.mat")
    names = mat_data["parameterNames"].reshape(-1,)
    names = [name for sublist in names for name in sublist]
    return names


def get_dataset(
    labels_file: str, params_file: str, names_file: str = "paremeterNames.mat"
) -> pd.DataFrame:
    labels = get_labels_mat(labels_file)
    parameters = get_parameters_mat(params_file)
    parameters.columns = get_parameter_names_mat(names_file)
    parameters["label"] = labels
    return parameters