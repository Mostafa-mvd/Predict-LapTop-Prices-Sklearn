
from sklearn.model_selection import train_test_split

import settings
import utils


if __name__ == "__main__":

    laptop_df = utils.get_dataframe(
        settings.csv_main_file_path, settings.all_column_names)

    ohe = utils.get_encoder()

    features_df = laptop_df[settings.feature_columns]

    encoded_features = utils.encode_features(ohe, features_df)

    target = laptop_df.Price_euros

    # split (= drop) our data in different groups for training and testing
    features_train, features_test, \
    target_train, target_test = train_test_split(
        encoded_features, target, test_size=10)

    dtc = utils.get_decision_tree_obj()

    utils.fit(features_train, target_train, dtc) # training

    y_predicted = dtc.predict(features_test)

    decoded_data = utils.decode_result(features_test, y_predicted, ohe)

    utils.csv_writer(decoded_data, settings.csv_predicted_file_path)
