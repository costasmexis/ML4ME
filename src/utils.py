import pandas as pd
from scipy.io import loadmat
from typing import Union


###### Function to load kinetic parameters related data from the .mat files
def get_labels_mat(filename) -> list:
    mat_data = loadmat(f"./data/{filename}")
    label = mat_data["class_vector_train"].reshape(
        -1,
    )
    label = [item for sublist in label for item in sublist]
    return label


def get_parameters_mat(filename) -> pd.DataFrame:
    mat_data = loadmat(f"./data/{filename}")
    parameters = mat_data["training_set"]
    return pd.DataFrame(parameters.T)


def get_parameter_names_mat(filename) -> list:
    mat_data = loadmat("./data/paremeterNames.mat")
    names = mat_data["parameterNames"].reshape(
        -1,
    )
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


###### Functions to load FFC related data from the .mat files
def get_cc_mat(filename: str):
    mat_data = loadmat(filename)
    enzyme = mat_data["commonEnzAct"][0][0][2][0][0][0]
    print(f"Enzyme: {enzyme}")

    commonEnz = mat_data["commonEnzAct"][0][0][3][0][0]
    commonEnz = [item for sublist in commonEnz for item in sublist]
    commonEnz = [item[0] for item in commonEnz]
    print(f"Common enzymes: {commonEnz}")

    allEnzymes = mat_data["commonEnzAct"][0][0][4][0][0]
    allEnzymes = [item for sublist in allEnzymes for item in sublist]
    allEnzymes = [item[0] for item in allEnzymes]
    print(f"Number of all enzymes: {len(allEnzymes)}")

    commonConCoeff = mat_data["commonEnzAct"][0][0][0][0][0]
    allConCoeff = mat_data["commonEnzAct"][0][0][1][0][0]

    commonConCoeff = pd.DataFrame(commonConCoeff[:, 0, :], columns=commonEnz)
    allConCoeff = pd.DataFrame(allConCoeff[:, 0, :], columns=allEnzymes)

    return enzyme, commonEnz, allEnzymes, commonConCoeff, allConCoeff
