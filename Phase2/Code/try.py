
import os
import cv2
import numpy as np
import random
import skimage
import PIL
import sys
import pandas as pd



def ReadLabels(LabelsPathTrain):
    # Check if file exists
    if not os.path.isfile(LabelsPathTrain):
        print(f"ERROR: Train Labels do not exist in {LabelsPathTrain}")
        sys.exit()
    
    # Read CSV file
    df = pd.read_csv(LabelsPathTrain)
    
    # Extract the last 8 columns (h1 to h8) as a numpy array
    TrainLabels = df.iloc[:, 2:].values
    
    return TrainLabels


LabelsPathTrain = '../Data/TrainSet/labels.csv'
labels = ReadLabels(LabelsPathTrain)

# Print shape and first few rows for verification
print("Shape of labels:", labels.shape)
print("\nFirst few rows:" ,len(labels))
print(labels[:3])



def ReadDirNames(ReadPath):
    """
    Inputs:
    ReadPath is the path of the CSV file you want to read
    Outputs:
    DirNames is a list of all patch_id values from the CSV file
    """
    # Check if the file exists
    if not os.path.isfile(ReadPath):
        print(f"ERROR: File does not exist at {ReadPath}")
        return None
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(ReadPath)
    
    # Extract the 'patch_id' column as a list
    DirNames = df['patch_id'].tolist()
    
    return DirNames

patch_ids = ReadDirNames(LabelsPathTrain)
    
    # Print the first few patch IDs for verification
if patch_ids:
    print("Number of patch IDs:", len(patch_ids))
    print("First few patch IDs:", patch_ids[:5])