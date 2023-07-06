from flex.data import Dataset
from flex.common import utils

def credit2(out_dir: str = "."):
    # TODO: Convert dataset to Categorical to use with an ID3 tree.
    # kaggle_path ='brycecf/give-me-some-credit-dataset'
    import os
    import pandas as pd
    if not os.path.exists(f"{out_dir}/credit2.csv"):
        path_to_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit' \
                                '%20card%20clients.xls '
        dataset = pd.read_excel(path_to_train, index_col=0)
        dataset = dataset.iloc[1:]
        dataset.to_csv(f"{out_dir}/credit2.csv")
    else:
        dataset = pd.read_csv(f"{out_dir}/credit2.csv", index_col=0)
    y_data = dataset['Y'].to_numpy()
    X_data = dataset.drop(columns=['Y'], axis=1).to_numpy()

    from sklearn.model_selection import train_test_split
    X_data, X_test, y_data, y_test = train_test_split(X_data, y_data, test_size=0.2)
    train_data_object = Dataset(X_data=X_data, y_data=y_data)
    test_data_object = Dataset(X_data=X_test, y_data=y_test)
    return train_data_object, test_data_object

def nursery(out_dir: str = '.', ret_feature_names: bool = False):
    # sourcery skip: assign-if-exp, extract-method, swap-if-expression
    """Function that load the nursery dataset from UCI database

    Args:
        out_dir (str, optional): Output dir. Defaults to '.'.
    """
    import os
    import pandas as pd
    col_names = ['parents', 'has_nurs', 'form', 'children', 'housin', 'finance', 
        'social', 'health', 'label']
    if not os.path.exists(f"{out_dir}/nursery.csv"):
        path_to_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"
        dataset = pd.read_csv(path_to_train, header=None)
        dataset.columns = col_names
        dataset['label'] = dataset['label'].astype('category')
        dataset['label'] = dataset['label'].cat.codes
        dataset.to_csv(f"{out_dir}/nursery.csv", index=False)
    else:
        dataset = pd.read_csv(f"{out_dir}/nursery.csv")
    y_data = dataset['label'].to_numpy()
    X_data = dataset.drop(columns=['label'], axis=1).to_numpy()

    from sklearn.model_selection import train_test_split
    X_data, X_test, y_data, y_test = train_test_split(X_data, y_data, test_size=0.2)
    train_data_object = Dataset(X_data=X_data, y_data=y_data)
    test_data_object = Dataset(X_data=X_test, y_data=y_test)
    col_names = [f"x{i}" for i in range(len(col_names))]
    if ret_feature_names:
        return train_data_object, test_data_object, col_names
    return train_data_object, test_data_object

def adult(out_dir: str = '.'):
    # TODO: Convert dataset to Categorical to use with an ID3 tree.
    """Function that load the adult dataset from the UCI database

    Args:
        out_dir (str, optional): Output dir to save the dataset. Defaults to '.'.
    """
    import os
    import pandas as pd
    if not os.path.exists(f"{out_dir}/adult_train.csv"):
        path_to_train = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        path_to_test = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
        x_columns = ['x' + str(i) for i in range(14)]
        y_column = 'label'
        train_data = pd.read_csv(path_to_train, names=x_columns + [y_column])
        test_data = pd.read_csv(path_to_test, names=x_columns + [y_column])
        test_data = test_data.iloc[1:]
        test_data['x0'] = [int(val) for val in test_data['x0']]

        # Preprocess the dataset
        def preprocess_adult_no_categorical(trainin_dataset):
            """Function to preprocess Adult dataset and transform it to a categorical dataset.
            Function assumes dataset has columns labels ['x0', 'x1', 'x2',...,'x13', 'target']
            The feature 'fnlwgt' ('x2') will be dropped if it is in the dataset.
            Args:
                training_dataset (pd.DataFrame): Adult DataFrame
            Returns:
                pd.DataFrame: Adult transformed into a categorical dataset.
            """
            from sklearn.feature_extraction import DictVectorizer
            dv = DictVectorizer()
            dv_data = dv.fit_transform([dict(row) for index, row in trainin_dataset.iterrows()])
            dv_data = pd.DataFrame(dv_data.toarray(), columns=dv.feature_names_)
            feature_types = ['int'] * len(dv.feature_names_)
            return dv_data, feature_types, dv.feature_names_

        train_labels = train_data.apply(lambda row: 1 if '>50K' in row['label'] else 0, axis=1).to_numpy()
        test_labels = test_data.apply(lambda row: 1 if '>50K' in row['label'] else 0, axis=1).to_numpy()
        train_data, feature_types, feature_names = preprocess_adult_no_categorical(train_data.drop(columns=['label'], axis=1))
        test_data, _, _ = preprocess_adult_no_categorical(test_data.drop(columns=['label'], axis=1))
        train_data['label'] = train_labels
        test_data['label'] = test_labels
        train_data.to_csv(f"{out_dir}/adult_train.csv", index=False)
        test_data.to_csv(f"{out_dir}/adult_test.csv", index=False)
    else:
        train_data = pd.read_csv(f"{out_dir}/adult_train.csv")
        test_data = pd.read_csv(f"{out_dir}/adult_test.csv")
    y_data = train_data['label'].to_numpy()
    X_data = train_data.drop(columns=['label'], axis=1).to_numpy()
    from sklearn.model_selection import train_test_split
    X_data, X_test, y_data, y_test = train_test_split(X_data, y_data, test_size=0.3)
    # y_test = test_data['label'].to_numpy()
    # X_test = test_data.drop(columns=['label'], axis=1).to_numpy()

    train_data_object = Dataset(X_data=X_data, y_data=y_data)
    test_data_object = Dataset(X_data=X_test, y_data=y_test)
    return train_data_object, test_data_object

def bank(out_dir: str = '.'):
    # TODO: Convert dataset to Categorical to use with an ID3 tree.
    """Function that load the Bank dataset from the UCI database.

    Args:
        out_dir (str, optional): _description_. Defaults to '.'.
    """
    import os
    import pandas as pd
    if not os.path.exists(f"{out_dir}/bank-full.csv"):
        raise FileNotFoundError(
            "Option not available right now. Please refer to the UCI repository and " + \
                "manually download the dataset."
        )
    else:
        dataset = pd.read_csv(f"{out_dir}/bank-full.csv", sep=';')
    ##### TODO: Pasar código al primer if una vez esté funcionando.
    x_columns = [f'x{str(i)}' for i in range(16)]
    y_column = 'label'
    cols = x_columns+[y_column]
    dataset = dataset.rename(columns=dict(zip(dataset.columns, cols)))
    dataset = dataset[(dataset.astype(str) != ' ?').all(axis=1)]
    dataset['label'] = dataset.apply(lambda row: 1 if 'yes' in row['label'] else 0, axis=1)
    dataset['x4'] = dataset.apply(lambda row: 1 if 'yes' in row['x4'] else 0, axis=1)
    dataset['x6'] = dataset.apply(lambda row: 1 if 'yes' in row['x6'] else 0, axis=1)
    dataset['x7'] = dataset.apply(lambda row: 1 if 'yes' in row['x7'] else 0, axis=1)
    dataset = dataset.drop(['x9', 'x10', 'x15'], axis=1)
    dataset = pd.get_dummies(dataset, columns=['x1', 'x2', 'x3', 'x8'])
    #####
    y_data = dataset['label'].to_numpy()
    X_data = dataset.drop(columns=['label'], axis=1).to_numpy()

    from sklearn.model_selection import train_test_split
    X_data, X_test, y_data, y_test = train_test_split(X_data, y_data, test_size=0.3)
    train_data_object = Dataset(X_data=X_data, y_data=y_data)
    test_data_object = Dataset(X_data=X_test, y_data=y_test)
    return train_data_object, test_data_object

def magic(out_dir: str = '.'):
    """Function that load the Bank dataset from the UCI database.

    Args:
        out_dir (str, optional): _description_. Defaults to '.'.
    """
    import os
    import pandas as pd
    if not os.path.exists(f"{out_dir}/magic.csv"):
        path_to_train = 'http://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data'
        col_names = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'label']
        dataset = pd.read_csv(path_to_train, names=col_names)
    else:
        dataset = pd.read_csv(f"{out_dir}/magic.csv", sep=';')
        col_names = list(dataset.columns)
    c = {'g':1, 'h':0}
    # SPLIT DATA INTRO TRAIN-VALIDATION
    y_data = dataset['target'].apply(lambda x:c[x]).to_numpy()
    X_data = dataset.drop(['target'], axis=1).to_numpy()

    from sklearn.model_selection import train_test_split
    X_data, X_test, y_data, y_test = train_test_split(X_data, y_data, test_size=0.3)
    train_data_object = Dataset(X_data=X_data, y_data=y_data)
    test_data_object = Dataset(X_data=X_test, y_data=y_test)
    return train_data_object, test_data_object
