'''
This file will serve as the main file to set up and run our analyses. First, you will choose the dataset you wish to use, and from there you will choose the type of model you want to use.
This will serve as a way for us to easily benchmark and test our different sources and the like as we go through the project.
'''

# Import Necessary Libraries
from ultralytics import YOLO
import pathlib, shutil, glob, os, torch, random
from yaspin import yaspin
from model_Utils import ModelUtils
import pandas as pd
# import kagglehub as kh
from PIL import Image, ImageDraw
import numpy as np

# Define data-options dict:
files = os.listdir()
data_options = {file for file in files}

# Import Necessary Dataset(s)
run = True
while run:
  model_choice = input('''
  Here you need to select the\033[1m model\033[0m you want to use. Here are the options:
  1. Pre-trained YOLO Model
  2. Use our Latest Model
  Enter the corrosponding number to select an option. \n
  ''').strip()

  if model_choice not in ['1', '2']:
    print("Try again - enter either '1' or '2'.")
    continue

  else:
    data_choice = input('''
    Here you need to select the \033[1mdata\033[0m you want to use. Here are the options:
    1. FishInv Dataset
    2. FishInv + Megafauna Dataset
    3. Deep Fish Dataset
    4. FishInv + Deep Fish
    5. Fish Inv + Megafauna + Deep Fish
    6. Bay Campus \n
    ''').strip()
    print(f'''
    Here you need to select the \033[1mdata\033[0m you want to use. Here are the options:
    {data_options}
    ''')

  #if data_choice not in ['1', '2', '3', '4', '5', '6']:
  if data_choice not in range(len(data_options)):
    print('Try again - enter one of the numbers in the list.')
    continue

  # Change run condition if reqs are satisfied
  run = False

  print(f'You have selected the \033[1m{model_choice}\033[0m model and the \033[1m{data_choice}\033[0m dataset.')
  print('Please wait while we set up the model and data...')
  # Set up the model and data
  try:
      utils = ModelUtils(model_choice, data_choice)
      model = utils.set_model()
      data = utils.set_data()
      print('Model and Data Set Up Complete!')
      print(f'Testing the {model_choice} model with the {data_choice} dataset.')
      utils.run()
      print('Testing Complete!')

  except Exception as e:
      print('There was an error with the training and/or testing. Please check the logs for more information.')
      print(f'Error: {e}')
      print('Exiting...')
      exit(1)


