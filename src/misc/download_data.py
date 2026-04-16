from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize the Kaggle API
api = KaggleApi()
api.authenticate()

# Download a dataset (replace with your dataset path)
dataset_name = "deekshith18/my-something-something-v2-data"  # Example dataset
api.dataset_download_files(dataset_name, path=".", unzip=True)
