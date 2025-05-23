# This program is going to be the helper function that takes two string inputs to determine which
# dataset and model to use. It will then run the training and testing of the model on the dataset.

# Import Libraries
import os
import shutil
import glob
import random
import torch
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

# First, edit the .yaml file to include the correct paths for the train, val, and test directories.
with open('datasets/FishInvSplit/data.yaml', 'r') as f:
  data = f.read()
data = data.replace('train: ../train/images', 'train: /content/FishInvSplit/train/images')
data = data.replace('val: ../val/images', 'val: /content/FishInvSplit/val/images')
data = data.replace('test: ../test/images', 'test: /content/FishInvSplit/test/images')
with open('datasets/FishInvSplit/data.yaml', 'w') as f:
  f.write(data)

# Next, edit the Ultralytics/settings.json to ensure that the dataset directory is correct.
with open('ultralytics/settings.json', 'r') as f:
  data = f.read()
data = data.replace('"dataset": "../datasets/FishInvSplit"', '"dataset": "/content/FishInvSplit"')
with open('ultralytics/settings.json', 'w') as f:
  f.write(data)

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

    def set_model(self):
        if self.model_choice == '1': # Pre-trained YOLO Model
            self.model = YOLO('yolov8n.pt') # Here we load pre-trained YOLO model
            
        elif self.model_choice == '2': # Use our latest model
            with open('code/data/trained models/best.pt'):
                self.model = YOLO('code/trained models/best.pt')
                # Here we load our latest model from Colab; add your own model path here

        else:
            raise ValueError("Invalid model choice")
        return self.model

    def set_data(self):
        if self.data_choice == '1':
            self.data = 'datasets/FishInvSplit/data.yaml'

        elif self.data_choice == '2':
            self.data = 'data/fishinv_megafauna.yaml'

        elif self.data_choice == '3':
            self.data = 'data/deepfish.yaml'

        elif self.data_choice == '4':
            self.data = 'data/fishinv_deepfish.yaml'

        elif self.data_choice == '5':
            self.data = 'data/fishinv_megafauna_deepfish.yaml'

        elif self.data_choice == '6':
            self.data = 'data/baycampus.yaml'

        else:
            raise ValueError("Invalid data choice")

        return self.data

    # Here we set the seed for reproducibility
    # This is important for ensuring that the results are consistent across different runs
    # Furthermore, we want to be able to explicitly set the seed for the model and data
    def set_seed(self, seed=42):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':32:8'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':64:8'
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

    def set_data_device(self):
        device = self.set_device()
        self.data.to(device)
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
        self.set_model()
        self.set_data()
        self.train_model()
        self.test_model()
        self.save_results()
        self.cleanup()
        return True