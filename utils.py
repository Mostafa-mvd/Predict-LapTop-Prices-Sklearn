import itertools
import pandas as pd
import settings

# Using for classification and regression problems
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder


def get_category_name(index, lengths):
    """" get category name from your features columns in settings modules"""

    counter = 0

    for i in range(len(lengths)):
        counter += lengths[i]
        if index < counter:
            return settings.feature_columns[i]

    raise ValueError(
        'The index is higher than the number of categorical values')


def decode_result(X, y, encoder):
    decoded_result = [{} for _ in range(len(y))]

    all_categories = list(itertools.chain(*encoder.categories_))

    category_lengths = [len(encoder.categories_[i])
                        for i in range(len(encoder.categories_))]

    encoded_rows, encoded_columns = X.nonzero()

    for row_index, feature_index in zip(encoded_rows, encoded_columns):
        category_value = all_categories[feature_index]

        category_name = get_category_name(
            feature_index, category_lengths)

        decoded_result[row_index][category_name] = category_value

        if not decoded_result[row_index].get('PredictedPriceEuro'):
            decoded_result[row_index]['PredictedPriceEuro'] = y[row_index]

    return decoded_result


def get_dataframe(file_path, columns):
    return pd.read_csv(
        filepath_or_buffer=file_path, header=None,
        names=columns, encoding='latin-1')


def get_encoder():
    return OneHotEncoder(handle_unknown="ignore")


def get_decision_tree_obj():
    return DecisionTreeClassifier()


def encode_features(encoder, features_dataframe):
    """encoding our features string format to integer"""

    return encoder.fit_transform(features_dataframe)


def fit(X_train, y_train, dtc_obj):
    # dtc is decision tree classifier object
    # Build a decision tree classifier from the training set (X, y).
    dtc_obj.fit(X_train, y_train)
