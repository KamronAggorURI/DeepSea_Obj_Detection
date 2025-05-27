'''
This file will serve as the main file to set up and run our analyses. First, you will choose the dataset you wish to use, and from there you will choose the type of model you want to use.
This will serve as a way for us to easily benchmark and test our different sources and the like as we go through the project.
'''

# Import Necessary Libraries
from ultralytics import YOLO
import os
from model_Utils import ModelUtils

# Define data-options dict:
files = os.listdir()
data_options = {file for file in files}

# # Import Necessary Dataset(s)
# run = True
# while run:
#   model_choice = input('''
#   Here you need to select the\033[1m model\033[0m you want to use. Here are the options:
#   1. Pre-trained YOLO Model
#   2. Use our Latest Model
#   Enter the corrosponding number to select an option. \n
#   ''').strip()

#   if model_choice not in ['1', '2']:
#     print("Try again - enter either '1' or '2'.")
#     continue

#   else:
#     data_choice = input('''
#     Here you need to select the \033[1mdata\033[0m you want to use. Here are the options:
#     1. FishInv Dataset
#     2. FishInv + Megafauna Dataset
#     3. Deep Fish Dataset
#     4. FishInv + Deep Fish
#     5. Fish Inv + Megafauna + Deep Fish
#     6. Bay Campus \n
#     ''').strip()
#     print(f'''
#     Here you need to select the \033[1mdata\033[0m you want to use. Here are the options:
#     {data_options}
#     ''')

# Define datasets directory and list the available datasets
datasets_dir = os.path.join(os.getcwd(), 'datasets')
if not os.path.exists(datasets_dir):
    print("No 'datasets' directory found. Please ensure you have the datasets in the correct location.")
    exit(1)

dataset_names = [d for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]
if not dataset_names:
  print("No datasets found in the 'datasets' directory. Please ensure you have the datasets in the correct location.")
  exit(1)

# Do the same for models directory
models_dir = os.path.join(os.getcwd(), 'models')
if not os.path.exists(models_dir):
    print("No 'models' directory found. Please ensure you have the models in the correct location.")
    exit(1)

model_names = [m for m in os.listdir(models_dir) if m.endswith('.pt')]
if not model_names:
  print("No models found in the 'models' directory. Please ensure you have the models in the correct location.")
  exit(1)

# Now we can ask the user to select a model and dataset
run = True
while run:
  print("Available datasets:")
  for i, dataset in enumerate(dataset_names, start=1):
      print(f"{i}. {dataset}")
  data_choice = input("\nSelect a dataset by entering the corresponding number: ").strip()
  
  print("Available models:")
  for i, model in enumerate(model_names, start=1):
      print(f"{i}. {model}")
  model_choice = input("\nSelect a model by entering the corresponding number: ").strip()
  
  data_choice = int(data_choice) - 1  # Convert to index
  model_choice = int(model_choice) - 1 
  data_choice = dataset_names[data_choice]
  model_choice = model_names[model_choice]

  # Check if the selected dataset and model exist
  if data_choice not in dataset_names:
      print(f"Dataset '{data_choice}' not found in the 'datasets' directory.")
      exit(1)
  if model_choice not in model_names:
      print(f"Model '{model_choice}' not found in the 'models' directory.")
      exit(1)

  # Change run condition if reqs are satisfied
  run = False

  print(f'You have selected the \033[1m{model_choice}\033[0m model and the \033[1m{data_choice}\033[0m dataset.')
  print('Please wait while we set up the model and data...')
  # Set up the model and data
  try:
      utils = ModelUtils(model_choice, data_choice)
      utils.yaml_config(os.path.join(datasets_dir, data_choice, 'data.yaml'))
      print('YAML Configuration Complete.')
      model = utils.set_model()
      data = utils.set_data()
      print('Model and Data Set Up Complete.')
      print(f'Testing the {model_choice} model with the {data_choice} dataset.')
      utils.run()
      print("Testing Complete. Results saved in the 'data' directory.")

  except Exception as e:
      print('There was an error with the training and/or testing. Please check the system logs for more information.')
      print(f'Error: {e}')
      print('Exiting...')
      exit(1)


