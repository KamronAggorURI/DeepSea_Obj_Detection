# This program is going to be the helper function that takes two string inputs to determine which
# dataset and model to use. It will then run the training and testing of the model on the dataset.

# Import Libraries
import os
# import tensorflow as tf # <- This is not used in the current code, but can be useful for loading datasets in cloud applications
# import tensorflow_datasets as tfds # <- This is not used in the current code, but can be useful for loading datasets in cloud applications
import shutil
import random
import torch
from ultralytics import YOLO
from pathlib import Path


# Define the ModelUtils class
# This class will handle the model and data selection, training, testing, and result saving.
# It will also handle the cleanup of the runs directory after training and testing.
class ModelUtils:
    def __init__(self, model_choice, data_choice):
        self.model_choice = model_choice
        self.data_choice = data_choice
        self.model = None
        self.data = None
        self.results = None

    def yaml_config(self, yaml_path):
        # update the yaml file w the correct paths for the train, validation, and test dirs
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"The specified YAML file does not exist: {yaml_path}")
        with open(yaml_path, 'r') as f:
            data = f.read()
        dataset_dir = os.path.dirname(yaml_path)
        data = data.replace('train: ../train/images', f'train: {os.path.join(dataset_dir)}/train/images')
        data = data.replace('val: ../val/images', f'val: {os.path.join(dataset_dir)}/val/images')
        data = data.replace('test: ../test/images', f'test: {os.path.join(dataset_dir)}/test/images')
        with open(yaml_path, 'w') as f:
            f.write(data)
        print(f"YAML configuration updated for {yaml_path}.")

    def set_model(self):
        model_path = os.path.join('models', self.model_choice)
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            raise FileNotFoundError(f"The specified model file does not exist: {model_path}")
        return self.model

    def set_data(self):
        data_yaml = os.path.join('datasets', self.data_choice, 'data.yaml')
        if os.path.exists(data_yaml):
            self.data = data_yaml
        else:
            raise FileNotFoundError(f"The specified data YAML file does not exist: {data_yaml}")
        return self.data

        # Here we load the data from the YAML file
        self.data = YOLO(self.data)
        # This will load the data from the YAML file and return it as a YOLO object
        print(f"Data loaded from {self.data}.")

        return self.data

    # Here we set the seed
    # This is important for ensuring that the results are consistent across different runs
    # Furthermore, we want to be able to explicitly set the seed for the model and data
    def set_seed(self, seed=42):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed) # Set the Python hash seed for reproducibility

        # Set the CUBLAS workspace config (NVIDIA GPU); the first value is the workspace size, and the second value is the number of threads
        # The CUBLAS workspace config is important for ensuring that the model runs efficiently on the GPU
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
        
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':32:8' # This is a more conservative setting that should work on most GPUs
        
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':64:8' # This is a more conservative setting that should work on most GPUs

        return True

    # Here we set the device for the model and data
    # This is important for ensuring that the model and data are on the same device
    # Furthermore, we want to be able to explicitly set the device for the model and data
    def set_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        return device

    # Here we set the model and data to the same device
    def set_model_device(self):
        device = self.set_device()
        self.model.to(device)
        return device

    # Here we use our data to train the model
    def train_model(self):
        print("Training Model...")
        self.model.train(data=self.data, epochs=10, imgsz=640, batch=16)
        print("✔ Training Complete")
        return self.model

    # Here we use our data to test and benchmark the model
    def test_model(self):
        print("Testing Model...")
        self.results = self.model.val(data=self.data, imgsz=640, batch=16)
        print("✔ Testing Complete")
        return self.results

    # Here we save the results of the model
    # This is important for ensuring that the results are saved in a consistent format
    # Furthermore, saving our PyTorch model is important for ensuring that the model can be reused in the future
    def save_results(self):
        print("Saving Results...")
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        self.results.save(results_dir)
        print("✔ Results Saved")
        return results_dir

    # Here is a train_test function that will do training, testing, and save the results
    def train_test(self):
        print("Training and Testing...")
        self.set_seed()
        self.set_model_device()
        self.set_data_device()
        if self.model_choice == '1' or self.model_choice == '2':
            self.train_model()  
        self.test_model()
        self.save_results()
        print("✔ Training and Testing Complete")
        return True

    # Here we cleanup the runs directory after training and testing, removing unnecessary files
    def cleanup(self):
        print("Cleaning Up...")
        shutil.rmtree("runs")
        print("✔ Cleanup Complete")
        return True

    # Here we run the entire process
    def run(self):
        self.set_seed()
        self.set_model_device()
        self.set_model()
        self.set_data()
        self.yaml_config(self.data)
        self.train_model()
        self.test_model()
        self.save_results()
        self.cleanup()
        return True
    