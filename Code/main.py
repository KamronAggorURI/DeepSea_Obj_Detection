'''
This file will serve as the main file to set up and run our analyses. First, you will choose the dataset you wish to use, and from there you will choose the type of model you want to use.
This will serve as a way for us to easily benchmark and test our different sources and the like as we go through the project.
'''

# Import Necessary Libraries
from ultralytics import YOLO
import pathlib, shutil, glob, os, torch, random
from yaspin import yaspin
from model_Utils import train_test
import pandas as pd

# Import Necessary Dataset(s)
run = True
while run:
  model_choice = input('''
  Here you need to select the \033[1m model \033[0m you want to use. Here are the options:
  1. FishInv Model
  2. Deep Fish Object Detection Model
  3. Train my own model
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
  
try:
  train_test
  
except:
  
  
