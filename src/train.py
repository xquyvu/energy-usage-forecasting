import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from azureml.core.run import Run
from azureml.core.workspace import Workspace


def load_data(dataset_name):
    return ws.datasets[dataset_name].to_pandas_dataframe()


def extract_features(dataframe):
    # Mark the period between sunrise and sunset
    for col in ['SUNRISE', 'SUNSET']:
        dataframe[col] = pd.to_datetime(dataframe['DATE'].dt.date) \
            + dataframe[col].apply(lambda x: pd.Timedelta(int(x.split(':')[0]), 'h')) \
            + dataframe[col].apply(lambda x: pd.Timedelta(int(x.split(':')[0]), 'm'))

    dataframe['SUN'] = np.logical_and(
            dataframe['SUNRISE'] < dataframe['DATE'],
            dataframe['DATE'] < dataframe['SUNSET']
    ).astype(int)

    # Lagged features
    dataframe['24hr_lag'] = dataframe['TOTAL Load'].shift(48)

    # Time features
    dataframe['DOW'] = dataframe['DATE'].dt.dayofweek
    dataframe['H'] = dataframe['DATE'].dt.hour
    dataframe['M'] = dataframe['DATE'].dt.month
    dataframe['weekend'] = (dataframe['DOW'] < 5).astype(int)

    # Drop unecessary columns
    dataframe = dataframe.drop(['DATE', 'SUNSET', 'SUNRISE'], axis=1)

    return dataframe


def clean_data(dataframe):
    dataframe = dataframe.dropna()

    features = dataframe.drop('TOTAL Load', axis=1)
    labels = dataframe['TOTAL Load']

    return features, labels


def load_and_clean(dataset_name):
    dataframe = load_data(dataset_name)
    dataframe = extract_features(dataframe)
    features, labels = clean_data(dataframe)

    return features, labels


try:
    # Get workspace if run locally
    ws = Workspace.from_config()
except:
    # Get workspace if run remotely
    ws = Run.get_context().experiment.workspace


# Run
run = Run.get_context()

# Load and clean data
features_train, labels_train = load_and_clean('energy-forecast-data-training')
features_val, labels_val = load_and_clean('energy-forecast-data-validation')


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_estimators', type=int, default=100,
        help="The number of trees in the forest."
    )
    parser.add_argument(
        '--max_depth', type=int, default=5,
        help="The maximum depth of the tree"
    )

    args = parser.parse_args()

    run.log("Number of estimators:", np.float(args.n_estimators))
    run.log("Max depth:", np.int(args.max_depth))

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    ).fit(features_train, labels_train)

    run.log('R2_valid', np.float(model.score(features_val, labels_val)))
    run.log('R2_train', np.float(model.score(features_train, labels_train)))

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/hyperdrive_model.joblib')


if __name__ == '__main__':
    main()
