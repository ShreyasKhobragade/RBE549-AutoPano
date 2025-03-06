#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW   
from Network.Network import HomographyModel, LossFn
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
import wandb
import kornia
import pandas as pd
import torch.nn.functional as F

wandb.init(
    # set the wandb project where this run will be logged
    project="Check"

    # # track hyperparameters and run metadata
    # config={
    # "learning_rate": 0.02,
    # "architecture": "CNN",
    # "dataset": "CIFAR-100",
    # "epochs": 10,
    # }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

def stn(image, H):
    batch = len(image)
    H_inv = torch.linalg.inv(H)
    y, x = torch.meshgrid(torch.linspace(-1, 1, 128),
                          torch.linspace(-1, 1, 128), indexing="ij")
    grid = torch.stack((x, y, torch.ones_like(x)), dim=-1).view(-1, 3).T
    grid = grid.repeat(batch, 1, 1).to(device)
    new_pts = torch.bmm(H_inv, grid).permute(0, 2, 1).to(device)
    new_pts = new_pts[:, :, :2] / new_pts[:, :, 2:].clamp(min=1e-8)
    new_grid = new_pts.view(batch, 128, 128, 2)
    warped_IA = F.grid_sample(image, new_grid, align_corners=True)

    return warped_IA

def TensorDLT(H4pt_batch, CA_batch):
    batch_size = H4pt_batch.shape[0]
    H4pt = H4pt_batch.reshape(batch_size, 4, 2)
    CA = CA_batch.reshape(batch_size, 4, 2)
    A = torch.zeros(batch_size, 8, 9, device=H4pt_batch.device)
    for b in range(batch_size):
        for i in range(4):
            x, y = CA[b, i]  # Source points
            u, v = H4pt[b, i] - CA[b, i]  # Offset from source
            A[b, 2*i] = torch.tensor([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
            A[b, 2*i + 1] = torch.tensor([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    _, _, Vt = torch.linalg.svd(A)
    H = Vt[..., -1].reshape(batch_size, 3, 3)
    H = H / H[:, 2:3, 2:3]
    return H
 

def ReadCorners(LabelsPathTrain):
    if not os.path.isfile(LabelsPathTrain):
        print(f"ERROR: Train Labels do not exist in {LabelsPathTrain}")
        sys.exit()
    df = pd.read_csv(LabelsPathTrain)
    TrainCorners = df.iloc[:, 10:].values
    return TrainCorners

def ReadImageNames(ReadPath):
    if not os.path.isfile(ReadPath):
        print(f"ERROR: File does not exist at {ReadPath}")
        return None
    df = pd.read_csv(ReadPath)
    ImageNames = df['image_id'].tolist()
    return ImageNames

def GenerateBatchUnSup(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize, T):
    I1Batch = []
    # CoordinatesBatch = []
    CornerBatch = []
    PABatch = []
    PBBatch = []
    IA = []
    if T == 0:
        name = "TrainSet"
        name1 = 'Train'
    else:
        name = "ValSet"
        name1 = 'Val'
    
    # LabelsPathTrain = '../Data/TrainSet/labels.csv'
    Corners = ReadCorners(BasePath + os.sep + name + "/labels.csv")
    ImageName = ReadImageNames(BasePath + os.sep + name + "/labels.csv")

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(DirNamesTrain) - 1)
        RandImageName_o = BasePath + os.sep + name + "/original_patches/" + DirNamesTrain[RandIdx] + "_original.jpg"
        RandImageName_w = BasePath + os.sep + name + "/warped_patches/" + DirNamesTrain[RandIdx] + "_warped.jpg"
        imgname = BasePath + os.sep + name1 + os.sep + ImageName[RandIdx]
        
        ImageNum += 1
        # DirNamesVal
        # Read images
        I1 = np.float32(cv2.imread(RandImageName_o, cv2.IMREAD_GRAYSCALE)) / 255.0
        I2 = np.float32(cv2.imread(RandImageName_w, cv2.IMREAD_GRAYSCALE)) / 255.0
        Image = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE) / 255.0
        Image = cv2.resize(Image, (320, 240))
        # Add new axis and concatenate
        I1 = I1[..., np.newaxis]
        I2 = I2[..., np.newaxis]
        IS = np.concatenate((I1, I2), axis=2)
        Image = Image[..., np.newaxis]

        # Convert NumPy arrays to PyTorch tensors (cast to float32)
        IS = torch.from_numpy(IS).permute(2, 0, 1).float()  # Shape: (C x H x W)
        I1 = torch.from_numpy(I1)
        # .permute(1, 0, 1).float()  # Shape: (C x H x W)
        I2 = torch.from_numpy(I2)
        # .permute(1, 0, 1).float()  # Shape: (C x H x W)
        Image = torch.from_numpy(Image).permute(2, 0, 1).float().to(device)
        Corner_A = torch.tensor(Corners[RandIdx], dtype=torch.float32)
        # Append tensors to batch
        I1Batch.append(IS)
        CornerBatch.append(Corner_A)
        PABatch.append(I1)
        PBBatch.append(I2)
        IA.append(Image)


    # Stack batches and move to device
    # return torch.stack(I1Batch).to(device), torch.stack(CornerBatch).to(device), torch.stack(PABatch).to(device), torch.stack(PBBatch).to(device)
    return torch.stack(I1Batch).to(device), torch.stack(CornerBatch).to(device), torch.stack(PABatch).to(device), torch.stack(PBBatch).to(device), torch.stack(IA).to(device)

def GenerateBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize, T):
    I1Batch = []
    CoordinatesBatch = []
    if T == 0:
        name = "TrainSet"
    else:
        name = "ValSet"

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(DirNamesTrain) - 1)
        RandImageName_o = BasePath + os.sep + name + "/original_patches/" + DirNamesTrain[RandIdx] + "_original.jpg"
        RandImageName_w = BasePath + os.sep + name + "/warped_patches/" + DirNamesTrain[RandIdx] + "_warped.jpg"
        ImageNum += 1
        # DirNamesVal
        # Read images
        I1 = np.float32(cv2.imread(RandImageName_o, cv2.IMREAD_GRAYSCALE)) / 255.0
        I2 = np.float32(cv2.imread(RandImageName_w, cv2.IMREAD_GRAYSCALE)) / 255.0

        # Add new axis and concatenate
        I1 = I1[..., np.newaxis]
        I2 = I2[..., np.newaxis]
        IS = np.concatenate((I1, I2), axis=2)

        # Convert NumPy arrays to PyTorch tensors (cast to float32)
        IS = torch.from_numpy(IS).permute(2, 0, 1).float()  # Shape: (C x H x W)
        Coordinates = torch.tensor(TrainCoordinates[RandIdx], dtype=torch.float32)  # Ensure float32

        # Append tensors to batch
        I1Batch.append(IS)
        CoordinatesBatch.append(Coordinates)

    # Stack batches and move to device
    return torch.stack(I1Batch).to(device), torch.stack(CoordinatesBatch).to(device)




def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)


def TrainOperation(
    DirNamesTrain,
    DirNamesVal,
    TrainCoordinates,
    ValCoordinates,
    NumTrainSamples,
    NumValSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    model = HomographyModel().to(device)

    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    # Optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    Optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=1000, gamma=0.99)
    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")


    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        model.train()
        train_loss = 0
        val_loss = 0
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            
            if ModelType == "UnSup":
                I1Batch, CornerBatch, PABatch, PBBatch, IA = GenerateBatchUnSup(
                BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize, 0
                )
                I1Batch = I1Batch.float()
                CornerBatch = CornerBatch.float()
                PABatch = PABatch.float()
                PBBatch = PBBatch.float()
                # Predict output with forward pass
                PredicatedH4pt = model(I1Batch)
                H = TensorDLT(PredicatedH4pt, CornerBatch)
                # PABatch_input = PABatch.permute(0, 3, 1, 2)
                PBBatch = PBBatch.permute(0, 3, 1, 2)
                #LossThisBatch = 0
                warped_PA = torch.zeros_like(PBBatch)
                corners = CornerBatch.reshape(-1, 4, 2)
                # print('before stn')
                warped_IA = stn(IA,H.to(device))


                x_min = torch.min(corners[..., 0], dim=1)[0]  # Shape: [32]
                x_max = torch.max(corners[..., 0], dim=1)[0]  # Shape: [32]
                y_min = torch.min(corners[..., 1], dim=1)[0]  # Shape: [32]
                y_max = torch.max(corners[..., 1], dim=1)[0]  # Shape: [32]

                # Ensure coordinates are within bounds
                x_min = torch.clamp(x_min, min=0, max=128)
                x_max = torch.clamp(x_max, min=0, max=128)
                y_min = torch.clamp(y_min, min=0, max=128)
                y_max = torch.clamp(y_max, min=0, max=128)

                # Create normalized coordinates for grid_sample
                y_coords = torch.linspace(-1, 1, 128, device=device)
                x_coords = torch.linspace(-1, 1, 128, device=device)
                grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
                grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                grid = grid.repeat(MiniBatchSize, 1, 1, 1)  # Shape: [32, 128, 128, 2]

                # Use grid_sample to extract all patches at once
                warped_PA = F.grid_sample(warped_IA, grid, align_corners=True)
                LossThisBatch = torch.nn.L1Loss()(warped_PA, PBBatch)
                print("Loss: ", LossThisBatch.item())
                Optimizer.zero_grad()
                H.requires_grad = True
                warped_PA.requires_grad = True
                LossThisBatch.requires_grad = True
                LossThisBatch.backward()

            if ModelType == "Sup":
                I1Batch, CoordinatesBatch = GenerateBatch(
                    BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize, 0
                )

                I1Batch = I1Batch.float()
                CoordinatesBatch = CoordinatesBatch.float()

                # Predict output with forward pass
                PredicatedCoordinatesBatch = model(I1Batch)
                LossThisBatch = LossFn(PredicatedCoordinatesBatch, CoordinatesBatch)

                Optimizer.zero_grad()
                LossThisBatch.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            Optimizer.step()
            scheduler.step()

            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName = (
                    CheckPointPath
                    + str(Epochs)
                    + "a"
                    + str(PerEpochCounter)
                    + "model.ckpt"
                )

                torch.save(
                    {
                        "epoch": Epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": Optimizer.state_dict(),
                        # "loss": LossThisBatch,
                    },
                    SaveName,
                )
                print("\n" + SaveName + " Model Saved...")

            if ModelType == "Sup":
                #esult = LossThisBatch
                result = model.validation_step(I1Batch, CoordinatesBatch)
                train_loss += result["val_loss"].item()

            if ModelType == "UnSup":
                result = LossThisBatch
                print('result:', result)
                train_loss += result.item()

            # Tensorboard
            # Writer.add_scalar(
            #     "LossEveryIter",
            #     result["val_loss"],
            #     Epochs * NumIterationsPerEpoch + PerEpochCounter,
            # )
            # If you don't flush the tensorboard doesn't update until a lot of iterations!
            Writer.flush()

        train_loss = train_loss / NumIterationsPerEpoch
        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        torch.save(
            {
                "epoch": Epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": Optimizer.state_dict(),
                # "loss": LossThisBatch,
            },
            SaveName,
        )

        #wandb.log({"loss": train_loss / NumIterationsPerEpoch})  
        print("Epoch: ", Epochs, "Training Loss: ", train_loss)

        model.eval()

        NumIterationsPerEpoch = int(NumValSamples / MiniBatchSize / DivTrain)
        with torch.no_grad():
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                if ModelType == "Sup":
                    I1Batch, CoordinatesBatch = GenerateBatch(
                        BasePath, DirNamesVal, ValCoordinates, ImageSize, MiniBatchSize, 1
                    )

                    I1Batch = I1Batch.float()
                    CoordinatesBatch = CoordinatesBatch.float()

                    # Predict output with forward pass
                    PredicatedCoordinatesBatch = model(I1Batch)
                    LossThisBatch = LossFn(PredicatedCoordinatesBatch, CoordinatesBatch)
                    result = model.validation_step(I1Batch, CoordinatesBatch)
                    # print("Validation Loss: ", result["val_loss"].item())
                    val_loss += result["val_loss"].item()

                if ModelType == "UnSup":
                    I1Batch, CornerBatch, PABatch, PBBatch, IA = GenerateBatchUnSup(
                        BasePath, DirNamesVal, ValCoordinates, ImageSize, MiniBatchSize, 1
                    )
                    I1Batch = I1Batch.float()
                    CornerBatch = CornerBatch.float()
                    PABatch = PABatch.float()
                    PBBatch = PBBatch.float()

                    # Predict output with forward pass
                    PredicatedH4pt = model(I1Batch)
                    H = TensorDLT(PredicatedH4pt, CornerBatch)
                    PABatch_input = PABatch.permute(0, 3, 1, 2)
                    LossThisBatch = 0
                    warped_PA = torch.zeros_like(PBBatch)
                    corners = CornerBatch.reshape(-1, 4, 2)
                    
                    warped_IA = stn(IA,H.to(device))


                    x_min = torch.min(corners[..., 0], dim=1)[0]  # Shape: [32]
                    x_max = torch.max(corners[..., 0], dim=1)[0]  # Shape: [32]
                    y_min = torch.min(corners[..., 1], dim=1)[0]  # Shape: [32]
                    y_max = torch.max(corners[..., 1], dim=1)[0]  # Shape: [32]

                    # Ensure coordinates are within bounds
                    x_min = torch.clamp(x_min, min=0, max=128)
                    x_max = torch.clamp(x_max, min=0, max=128)
                    y_min = torch.clamp(y_min, min=0, max=128)
                    y_max = torch.clamp(y_max, min=0, max=128)

                    # Create normalized coordinates for grid_sample
                    y_coords = torch.linspace(-1, 1, 128, device=device)
                    x_coords = torch.linspace(-1, 1, 128, device=device)
                    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
                    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                    grid = grid.repeat(MiniBatchSize, 1, 1, 1)  # Shape: [32, 128, 128, 2]

                    # Use grid_sample to extract all patches at once
                    warped_PA = F.grid_sample(warped_IA, grid, align_corners=True)
                    LossThisBatch = torch.nn.L1Loss()(warped_PA, PBBatch)
                    # print("Validation Loss: ", LossThisBatch.item())
                    val_loss += LossThisBatch.item()

        print("\n" + SaveName + " Model Saved...")
        val_loss = val_loss / NumIterationsPerEpoch

        wandb.log({"Training loss": train_loss, "Validation loss": val_loss, "Epoch": Epochs})
        print("Epoch: ", Epochs, "Validation Loss: ", val_loss)

    wandb.finish()  


def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="../Data",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="../Checkpoints_UnSup_0005/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="UnSup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=50,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=32,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        DirNamesVal,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        NumValSamples,
        TrainCoordinates,
        ValCoordinates,
        NumClasses
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrain,
        DirNamesVal,
        TrainCoordinates,
        ValCoordinates,
        NumTrainSamples,
        NumValSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType,
    )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main() 