# This program is going to be the helper function that takes two string inputs to determine which
# dataset and model to use. It will then run the training and testing of the model on the dataset.

# Import Libraries
import os
# import tensorflow as tf # <- This is not used in the current code, but can be useful for loading datasets in cloud applications
# import tensorflow_datasets as tfds # <- This is not used in the current code, but can be useful for loading datasets in cloud applications
import shutil
import glob
import random
import torch
from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageEnhance

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

    def check_and_structure_dataset(self, dataset_path):
        """
        Checks if the dataset at dataset_path is structured for YOLO (train/val/test with images/labels).
        If not, attempts to restructure it.
        """
        required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
        missing = []
        for d in required_dirs:
            if not os.path.isdir(os.path.join(dataset_path, d)):
                missing.append(d)
        if not missing:
            print(f"Dataset at {dataset_path} is already structured.")
            return True

        print(f"Dataset missing directories: {missing}. Attempting to restructure...")

        # Try to move all images to train/images and all labels to train/labels IF flat
        image_exts = ('*.jpg', '*.jpeg', '*.png')
        label_exts = ('*.txt',)
        images = []
        for ext in image_exts:
            images.extend(glob.glob(os.path.join(dataset_path, ext)))
        labels = []
        for ext in label_exts:
            labels.extend(glob.glob(os.path.join(dataset_path, ext)))

        # Create directories if they don't exist
        os.makedirs(os.path.join(dataset_path, 'train/images'), exist_ok=True)
        os.makedirs(os.path.join(dataset_path, 'train/labels'), exist_ok=True)

        # Move images and labels
        for img in images:
            shutil.move(img, os.path.join(dataset_path, 'train/images', os.path.basename(img)))
        for lab in labels:
            shutil.move(lab, os.path.join(dataset_path, 'train/labels', os.path.basename(lab)))

        print(f"Moved {len(images)} images and {len(labels)} labels to train/.")

        # Optionally, split into train/val/test here if needed
        # Check again
        for d in required_dirs:
            if not os.path.isdir(os.path.join(dataset_path, d)):
                print(f"Warning: Directory {d} still missing after restructuring.")
                return False

        print(f"Dataset at {dataset_path} structured successfully.")
        return True

        # Create the necessary directories if they do not exist
        for subdir in ['train', 'val', 'test']:
            subdir_path = os.path.join(dataset_path, subdir, 'images')
            os.makedirs(subdir_path, exist_ok=True)

        print(f"Dataset structured at {dataset_path}.")
        return dataset_path

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

        return True

    # Here we set the device for the model and data
    # This is important for ensuring that the model and data are on the same device
    # Furthermore, we want to be able to explicitly set the device for the model and data
    def set_device(self):
        # Prefer CUDA if available, then MPS (Apple Silicon), then DirectML (Windows), else CPU
        if torch.cuda.is_available():
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True
                device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Apple Metal Performance Shaders (MPS) for Apple Silicon
            device = torch.device('mps')
        elif hasattr(torch.backends, 'directml') and torch.backends.directml.is_available():
            # DirectML for Windows (experimental in PyTorch)
            device = torch.device('directml')
        else:
            # Default to CPU
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
        print("Training Complete.")
        return self.model

    # Here we use our data to test and benchmark the model
    def test_model(self):
        print("Testing Model...")
        self.results = self.model.val(data=self.data, imgsz=640, batch=16)
        print("Testing Complete.")
        return self.results

    # Here we save the results of the model
    # This is important for ensuring that the results are saved in a consistent format
    # Furthermore, saving our PyTorch model is important for ensuring that the model can be reused in the future
    def save_results(self):
        print("Saving Results...")
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        self.results.save(results_dir)
        print("Results Saved.")
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
        print("Training and Testing Complete.")
        return True

    # Here we cleanup the runs directory after training and testing, removing unnecessary files
    def cleanup(self):
        print("Cleaning Up...")
        shutil.rmtree("runs")
        print("Cleanup Complete.")
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
    
# This class handles preprocessing of the model and data before training and testing
class ModelPreprocessing:
    def __init__(self, model_choice, data_choice):
        self.model_choice = model_choice
        self.data_choice = data_choice

    def crop(self, image, crop_size):
        # Crop the image to the specified size
        return image.crop((0, 0, crop_size[0], crop_size[1]))

    def resize(self, image, size):
        # Resize the image to the specified size
        return image.resize(size, resample=Image.BILINEAR)

    def normalize(self, image):
        # Normalize the image to the range [0, 1]
        return image / 255.0

    def greyscale(self, image):
        # Convert the image to grayscale
        return image.convert('L')

    def flip(self, image):
        return image.transpose(Image.FLIP_LEFT_RIGHT)

    def rotate(self, image, angle):
        # Rotate the image by the specified angle
        return image.rotate(angle, expand=True)

    def brightness(self, image, factor):
        # Adjust the brightness of the image
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def contrast(self, image, factor):
        # Adjust the contrast of the image
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def saturation(self, image, factor):
        # Adjust the saturation of the image
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    def hue(self, image, factor):
        # Adjust the hue of the image
        # Note: PIL does not have a built-in method for hue adjustment, so we convert to HSV and adjust
        hsv_image = image.convert('HSV')
        h, s, v = hsv_image.split()
        h = h.point(lambda p: (p + factor) % 256)
        return Image.merge('HSV', (h, s, v)).convert('RGB')

    def rand_augment(self, image):
        # This will apply a random augmentation to an image
        augmentations = [
            lambda img: self.crop(img, (224, 224)),
            lambda img: self.resize(img, (224, 224)),
            lambda img: self.normalize(img),
            lambda img: self.greyscale(img),
            lambda img: self.flip(img),
            lambda img: self.rotate(img, random.randint(0, 360)),
            lambda img: self.brightness(img, random.uniform(0.5, 1.5)),
            lambda img: self.contrast(img, random.uniform(0.5, 1.5)),
            lambda img: self.saturation(img, random.uniform(0.5, 1.5)),
            lambda img: self.hue(img, random.randint(-30, 30))
        ]
        augmentation = random.choice(augmentations)
        return augmentation(image)

    def preprocess_image(self, image_path):
        # Load the image
        image = Image.open(image_path).convert('RGB')
        # Apply random augmentation
        image = self.rand_augment(image)
        return image

    def preprocess_dataset(self, dataset_path):
        # Preprocess all images in the dataset
        image_paths = glob.glob(os.path.join(dataset_path, 'train/images', '*.jpg')) + \
                      glob.glob(os.path.join(dataset_path, 'train/images', '*.png'))
        for image_path in image_paths:
            image = self.preprocess_image(image_path)
            # Save the preprocessed image back to the same path
            image.save(image_path)
        print(f"Preprocessing complete for dataset at {dataset_path}.")
        return True

    def run(self):
        # Run the preprocessing on the dataset
        dataset_path = os.path.join('datasets', self.data_choice)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"The specified dataset path does not exist: {dataset_path}")
        self.preprocess_dataset(dataset_path)
        print("Model preprocessing complete.")
        return True
    
    