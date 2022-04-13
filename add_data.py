from clearml import Dataset

dataset = Dataset.create(dataset_name='imgs+joints',
                         dataset_project='dqn test')
dataset.add_files(path="./train_data")
dataset.upload()
dataset.finalize(verbose=True)
