from azureml.core import Workspace, Dataset


def register_dataset(dataset_type):
    # Create datasets from datastore
    dataset = Dataset.Tabular.from_delimited_files(
        [(datastore, 'data/{}.csv'.format(dataset_type))]
    )

    # Register
    dataset = dataset.register(
        workspace=ws,
        name='energy-forecast-data-{}'.format(dataset_type),
        description='{} set'.format(dataset_type)
    )

    print(dataset)


ws = Workspace.from_config()

# get the datastore to upload prepared data
datastore = ws.get_default_datastore()

# upload the local file from src_dir to the target_path in datastore
datastore.upload(src_dir='data', target_path='data')

for dataset_type in ['training', 'validation']:
    register_dataset(dataset_type)
