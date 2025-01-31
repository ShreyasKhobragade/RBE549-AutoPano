"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import os
import cv2
import numpy as np
import random
import skimage
import PIL
import sys
import pandas as pd

# Don't generate pyc codes
sys.dont_write_bytecode = True


def SetupAll(BasePath, CheckPointPath):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    CheckPointPath - Path to save checkpoints/model
    Outputs:
    DirNamesTrain - Variable with Subfolder paths to train files
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    ImageSize - Size of the image
    NumTrainSamples - length(Train)
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize
    Trainabels - Labels corresponding to Train
    NumClasses - Number of classes
    """
    # Setup DirNames
    DirNamesTrain = SetupDirNames(BasePath, 'Train')
    DirNamesVal = SetupDirNames(BasePath, 'Val')

    # Read and Setup Labels
    # LabelsPathTrain = './TxtFiles/LabelsTrain.txt'
    LabelsPathTrain = '../Data/TrainSet/labels.csv'
    TrainLabels = ReadLabels(LabelsPathTrain)

    LabelsPathVal = '../Data/ValSet/labels.csv'
    ValLabels = ReadLabels(LabelsPathVal)

    # If CheckPointPath doesn't exist make the path
    if not (os.path.isdir(CheckPointPath)):
        os.makedirs(CheckPointPath)

    # Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    SaveCheckPoint = 100
    # Number of passes of Val data with MiniBatchSize
    NumTestRunsPerEpoch = 5

    # Image Input Shape
    ImageSize = [128, 128, 2]
    NumTrainSamples = len(TrainLabels)

    NumValSamples = len(ValLabels)

    # Number of classes
    NumClasses = 8

    return (
        DirNamesTrain,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainLabels,
        NumClasses,
    )
# I have made the changes in these file but nothing is getting returned from here to Train.py
# Will need to make changes in this file and Train file for the same


# def ReadLabels(LabelsPathTrain):
#     if not (os.path.isfile(LabelsPathTrain)):
#         print("ERROR: Train Labels do not exist in " + LabelsPathTrain)
#         sys.exit()
#     else:
#         TrainLabels = open(LabelsPathTrain, "r")
#         TrainLabels = TrainLabels.read()
#         TrainLabels = list(map(float, TrainLabels.split()))

#     return TrainLabels

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


def SetupDirNames(BasePath, Type):
    """
    Inputs:
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    Writes a file ./TxtFiles/DirNames.txt with full path to all image files without extension
    """
    if Type == 'Train':
        DirNamesTrain = ReadDirNames('../Data/TrainSet/labels.csv')
        return DirNamesTrain
    if Type == 'Val':
        DirNamesVal = ReadDirNames('../Data/ValSet/labels.csv')
        return DirNamesVal


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
