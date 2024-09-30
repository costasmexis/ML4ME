import pandas as pd
from scipy.io import loadmat
from typing import Union


###### Function to load kinetic parameters related data from the .mat files
def get_labels_mat(filename) -> list:
    mat_data = loadmat(filename)
    label = mat_data["class_vector_train"].reshape(
        -1,
    )
    label = [item for sublist in label for item in sublist]
    return label


def get_parameters_mat(filename) -> pd.DataFrame:
    mat_data = loadmat(filename)
    parameters = mat_data["training_set"]
    return pd.DataFrame(parameters.T)


def get_parameter_names_mat(filename) -> list:
    mat_data = loadmat(filename)
    names = mat_data["parameterNames"].reshape(-1,)
    names = [name for sublist in names for name in sublist]
    return names


def get_dataset(
    labels_file: str, params_file: str, names_file: str
) -> pd.DataFrame:
    labels = get_labels_mat(labels_file)
    parameters = get_parameters_mat(params_file)
    parameters.columns = get_parameter_names_mat(names_file)
    parameters["label"] = labels
    # In 'label' map 's' to 1 and 'ns' to 0
    parameters['label'] = parameters['label'].map({'s': 1, 'ns': 0})
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

##### Non-stratify split function for the dataset ######
def non_stratify_split(
    data: pd.DataFrame, train_size: float, target: str
) -> Union[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    X = data.drop(target, axis=1)
    y = data[target]
    
    # Separate the indices of the two classes
    indices_0 = y[y == 0].index
    indices_1 = y[y == 1].index

    # Calculate the number of samples for each class in the training set
    num_train_0 = int(train_size * 0.65)
    num_train_1 = int(train_size - num_train_0)

    # Select the training indices
    train_indices_0 = indices_0[:num_train_0]
    train_indices_1 = indices_1[:num_train_1]

    # Combine the training indices
    train_indices = train_indices_0.union(train_indices_1)

    # Select the test indices
    test_indices = indices_0[num_train_0:].union(indices_1[num_train_1:])

    # Split the data
    X_train = X.loc[train_indices]
    y_train = y.loc[train_indices]
    X_test = X.loc[test_indices]
    y_test = y.loc[test_indices]

    print(f'Traininig set shape: {X_train.shape}')
    print(f'Test set shape: {X_test.shape}')
    
    return X_train, X_test, y_train, y_test