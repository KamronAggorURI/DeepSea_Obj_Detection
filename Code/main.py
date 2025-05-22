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
import kagglehub as kh
from PIL import Image, ImageDraw


# Import Necessary Dataset(s)
run = True
while run:
  model_choice = input('''
  Here you need to select the \033[1m model \033[0m you want to use. Here are the options:
  1. Pre-trained YOLOv11n Model
  2. Use our Latest Model
  3. Train a new model
  Enter the corrosponding number to select an option. \n''' )

  if model_choice != (1 or 2 or 3):
    print("Try again - enter either '1' '2' or '3'.")

  else:
    data_choice = input('''
    Here you need to select the \033[1m data \033[0m you want to use. Here are the options:
    1. FishInv Dataset
    2. FishInv + Megafauna Dataset
    3. Deep Fish Dataset
    4. FishInv + Deep Fish
    5. Fish Inv + Megafauna + Deep Fish
    6. Bay Campus Dataset
    ''')

  if data_choice != (1 or 2 or 3 or 4 or 5 or 6):
    print('Try again - enter one of the numbers in the list.')

  else:
    run = False
    print(f'You have selected the \033[1m {model_choice} \033[0m model and the \033[1m {data_choice} \033[0m dataset.')
    with yaspin(text='Setting up model and data...', color='blue') as spinner:
      # Set up the model and data
      if model_choice == (1 or 2):
        model = ModelUtils.set_model(model_choice)
        data = ModelUtils.set_data(data_choice)
        print('Model and Data Set Up Complete!')
        print(f'Testing the {model_choice} model with the {data_choice} dataset.')
        try:
          ModelUtils.run()
          print('Testing Complete!')
    
        except:
          print('There was an error with the training and/or testing. Please check the logs for more information.')
          print('Exiting...')
          exit(1)

      elif model_choice == 3:
        model = ModelUtils.set_model(model_choice)
        data = ModelUtils.set_data(data_choice)
        print('Model and Data Set Up Complete!')
        print(f'Training and testing the {model_choice} model with the {data_choice} dataset.')
        try:
          ModelUtils.run()
          print('Training and testing Complete!')
    
        except:
          print('There was an error with the training and/or testing. Please check the logs for more information.')
          print('Exiting...')
          exit(1)


      if model_choice == ():
        try:
          ModelUtils.train_test(model_choice, data_choice)
          print('Training and Testing Complete!')
    
        except:
          print('There was an error with the training and testing. Please check the logs for more information.')
          print('Exiting...')
          exit(1)
  

