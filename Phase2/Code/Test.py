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

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import argparse
from Network.Network import HomographyModel, LossFn
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
import pandas as pd
import csv
import wandb
import kornia




wandb.init(
    # set the wandb project where this run will be logged
    project="AutoPano Phase2 Test"
    # }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_patches(image, num_patches=10, patch_size=(128, 128), rho=32, edge_margin=40, max_attempts=40, resize=(320, 240)):
    all_patches = []
    all_labels = []
    all_corners = []
    
    # Convert image to grayscale and resize
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, resize)
    
    height, width = image.shape[:2]
    patches_generated = 0
    attempts = 0
    
    while patches_generated < num_patches and attempts < max_attempts:
        try:
            attempts += 1
            
            # Get random patch coordinates with safe margins
            x = np.random.randint(edge_margin, width - patch_size[0] - edge_margin)
            y = np.random.randint(edge_margin, height - patch_size[1] - edge_margin)

            # Define original corners
            corners_A = np.float32([
                [x, y],
                [x + patch_size[0], y],
                [x, y + patch_size[1]],
                [x + patch_size[0], y + patch_size[1]]
            ])

            # Perturb corners randomly within [-rho, rho]
            perturb = np.random.randint(-rho, rho, size=(4, 2)).astype(np.float32)
            # perturb += np.random.randint(0,8, size=(1, 2))
            corners_B = corners_A + perturb

            # Verify corners are within image bounds
            if not (np.all(corners_B >= 0) and 
                   np.all(corners_B[:, 0] < width) and 
                   np.all(corners_B[:, 1] < height)):
                continue

            # Calculate homography
            H = cv2.getPerspectiveTransform(corners_A, corners_B)

            # Extract patches
            patch_A = image[y:y + patch_size[1], x:x + patch_size[0]]
            warped_image = cv2.warpPerspective(image, np.linalg.inv(H), (width, height))
            patch_B = warped_image[y:y + patch_size[1], x:x + patch_size[0]]

            # Verify patch dimensions and content
            if patch_A.shape != patch_size or patch_B.shape != patch_size:
                continue
            
            # Check if patches are valid (not empty or all black)
            if np.mean(patch_A) < 5 or np.mean(patch_B) < 5:
                continue

            # Stack patches
            patch_A = np.expand_dims(patch_A, axis=2)
            patch_B = np.expand_dims(patch_B, axis=2)
            input_stack = np.concatenate((patch_A, patch_B), axis=2)

            # Calculate 4-point homography (labels)
            p = corners_B - corners_A
            H4pt = p.flatten()

            all_patches.append(input_stack)
            all_labels.append(H4pt)
            all_corners.append(corners_A.flatten())
            patches_generated += 1

        except Exception as e:
            continue

    if patches_generated < num_patches:
        return None, None
    
    return all_patches, all_labels, all_corners

def process_dataset(input_folder, output_folder, num_patches=10, patch_size=(128, 128), 
                   rho=32, edge_margin=40, resize=(320, 240)):
    # Create output directories
    original_dir = os.path.join(output_folder, 'original_patches')
    warped_dir = os.path.join(output_folder, 'warped_patches')
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(warped_dir, exist_ok=True)

    # Create labels CSV file
    labels_file = os.path.join(output_folder, 'labels.csv')
    all_labels = []  # Collect all labels here

    # Process each image in the folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
    total_images = len(image_files)
    successful_images = 0
    
    print(f"Found {total_images} images to process")

    for idx, image_name in enumerate(image_files):
        image_path = os.path.join(input_folder, image_name)
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Skipping invalid image: {image_name}")
                continue

            # Generate patches and labels
            patches, labels, corners_list = generate_patches(image, num_patches, patch_size,
                                                              rho=rho,
                                                              edge_margin=edge_margin,
                                                              resize=resize)
            
            if patches is None:
                print(f"Failed to generate {num_patches} patches for {image_name}")
                continue

            # Save patches and collect labels
            for i, (patch, label, corners) in enumerate(zip(patches, labels, corners_list)):
                # Split the 2-channel patch into original and warped
                original = patch[:, :, 0]
                warped = patch[:, :, 1]

                # Save images
                patch_id = f'{os.path.splitext(image_name)[0]}_patch_{i:01d}'
                original_path = os.path.join(original_dir, f'{patch_id}_original.jpg')
                warped_path = os.path.join(warped_dir, f'{patch_id}_warped.jpg')

                # Save as grayscale images
                cv2.imwrite(original_path, original)
                cv2.imwrite(warped_path, warped)
                
                # Append labels to list (include both h4pt and corners_A)
                all_labels.append([image_name] + [patch_id] + label.tolist() + corners.tolist())
            
            successful_images += 1
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx+1}/{total_images} images. Successfully generated patches for {successful_images} images")

        except Exception as e:
            print(f"Error processing image {image_name}: {e}")
            continue

    # Write all collected labels to the CSV file at once
    with open(labels_file,'w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['image_id','patch_id','h1','h2','h3','h4','h5','h6','h7','h8',
                         'x1','y1','x2','y2','x3','y3','x4','y4'])
        writer.writerows(sorted(all_labels)) 

    print(f"\nProcessing complete!")
    print(f"Total images processed: {total_images}")
    print(f"Successfully generated patches for: {successful_images} images")
    print(f"Total patches generated: {successful_images * num_patches}")


# Don't generate pyc codes
sys.dont_write_bytecode = True


def ReadDirNames(ReadPath):
    # Check if the file exists
    if not os.path.isfile(ReadPath):
        print(f"ERROR: File does not exist at {ReadPath}")
        return None
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(ReadPath)
    
    # Extract the 'patch_id' column as a list
    DirNames = df['patch_id'].tolist()
    
    return DirNames

def SetupAll(BasePath):
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
    DirNamesTest = ReadDirNames('../Data/P1Ph2TestSet/TestSet/labels.csv')

    # Read and Setup Labels
    # LabelsPathTrain = './TxtFiles/LabelsTrain.txt'
    LabelsPathTest = '../Data/P1Ph2TestSet/TestSet/labels.csv'
    TestLabels = ReadLabels(LabelsPathTest)


    # Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch

    # Number of passes of Val data with MiniBatchSize
    NumTestRunsPerEpoch = 5

    # Image Input Shape
    ImageSize = [128, 128, 2]
    NumTestSamples = len(TestLabels)


    return (
        DirNamesTest,
        ImageSize,
        NumTestSamples,
        TestLabels,
    )

def StandardizeInputs(Img):
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################
    return Img

def ReadImages(Img):
    """
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    I1 = Img

    if I1 is None:
        # OpenCV returns empty list if image is not read!
        print("ERROR: Image I1 cannot be read")
        sys.exit()

    I1S = StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1

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
 
def GenerateBatch(BasePath, DirNamesTrain, TrainCoordinates, ImageSize, MiniBatchSize):
    I1Batch = []
    CoordinatesBatch = []
    CornerBatch = []
    PABatch = []
    PBBatch = []
    IA = []

    name = "TestSet"
    name1 = 'Phase2'

    
    # LabelsPathTrain = '../Data/TrainSet/labels.csv'
    Corners = ReadCorners(BasePath + os.sep + name + "/labels.csv")
    ImageName = ReadImageNames(BasePath + os.sep + name + "/labels.csv")

    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(DirNamesTrain) - 1)
        RandImageName_o = BasePath + os.sep + name + "/original_patches/" + DirNamesTrain[RandIdx] + "_original.jpg"
        RandImageName_w = BasePath + os.sep + name + "/warped_patches/" + DirNamesTrain[RandIdx] + "_warped.jpg"
        imgname = BasePath + name1 + os.sep + ImageName[RandIdx]
        
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
        Coordinates = torch.tensor(TrainCoordinates[RandIdx], dtype=torch.float32)  # Ensure float32
        CoordinatesBatch.append(Coordinates)

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
    return torch.stack(I1Batch).to(device), torch.stack(CoordinatesBatch).to(device), torch.stack(CornerBatch).to(device), torch.stack(PABatch).to(device), torch.stack(PBBatch).to(device), torch.stack(IA).to(device)

def epevalue(tensor1, tensor2):
    tensor1 = tensor1.to(tensor2.device)
    tensor1 = tensor1.view(tensor1.size(0), -1, 2)
    tensor2 = tensor2.view(tensor2.size(0), -1, 2)
    distances = torch.norm(tensor1 - tensor2, dim=2)
    epe = distances.mean()
    return epe


def TestOperation(BasePath, DirNamesTest, ImageSize, NumTestSamples, TestLabels, ModelType, ModelPath):
    # Predict output with forward pass, MiniBatchSize for Test is 1
    model = HomographyModel().to(device)

    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint["model_state_dict"])
    print(
        "Number of parameters in this model are %d " % len(model.state_dict().items())
    )

    model.eval()
    test_loss = 0
    epe = 0
    NumIterationsPerEpoch = int(NumTestSamples / 1)
    with torch.no_grad():
        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            if ModelType == "Sup":
                I1Batch, CoordinatesBatch, CornerBatch, PABatch, PBBatch, IA = GenerateBatch(
                    BasePath, DirNamesTest, TestLabels, ImageSize, 1
                )

                I1Batch = I1Batch.float()
                CoordinatesBatch = CoordinatesBatch.float()
                CornerBatch = CornerBatch.float()

                # Predict output with forward pass
                PredicatedH4pt = model(I1Batch)
                pred_corners = PredicatedH4pt + CornerBatch
                groundtruth = CoordinatesBatch + CornerBatch
                sepe = epevalue(pred_corners, groundtruth).item()
                LossThisBatch = LossFn(PredicatedH4pt, CoordinatesBatch)
                result = model.validation_step(I1Batch, CoordinatesBatch)
                print("Test Loss: ", result["val_loss"].item())
                wandb.log({"Testing loss": result["val_loss"].item(), "Image": PerEpochCounter})
                test_loss += result["val_loss"].item()
                epe += sepe

            if ModelType == "UnSup":
                I1Batch, CoordinatesBatch, CornerBatch, PABatch, PBBatch, IA = GenerateBatch(
                    BasePath, DirNamesTest, TestLabels, ImageSize, 1
                )
                I1Batch = I1Batch.float()
                CoordinatesBatch = CoordinatesBatch.float()
                CornerBatch = CornerBatch.float()
                PABatch = PABatch.float()
                PBBatch = PBBatch.float()
                PBBatch = PBBatch.permute(0, 3, 1, 2)

                # Predict output with forward pass
                PredicatedH4pt = model(I1Batch)
                pred_corners = PredicatedH4pt + CornerBatch
                groundtruth = CoordinatesBatch + CornerBatch
                uepe = epevalue(pred_corners, groundtruth).item()
                H = TensorDLT(PredicatedH4pt, CornerBatch)
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
                grid = grid.repeat(1, 1, 1, 1)  # Shape: [32, 128, 128, 2]

                # Use grid_sample to extract all patches at once
                warped_PA = F.grid_sample(warped_IA, grid, align_corners=True)
                
                LossThisBatch = torch.nn.L1Loss()(warped_PA, PBBatch)
                # print("Validation Loss: ", LossThisBatch.item())
                print("Test Loss: ", LossThisBatch.item())
                wandb.log({"Testing loss": LossThisBatch.item(), "Image": PerEpochCounter})
                test_loss += LossThisBatch.item()
                epe += uepe

    epe = epe / NumIterationsPerEpoch
    test_loss = test_loss / NumIterationsPerEpoch
    print("Testing Loss on the Test Set is ", test_loss)
    print("Average EPE Loss on the Test Set is ", epe)


def Accuracy(Pred, GT):
    """
    Inputs:
    Pred are the predicted labels
    GT are the ground truth labels
    Outputs:
    Accuracy in percentage
    """
    return np.sum(np.array(Pred) == np.array(GT)) * 100.0 / len(Pred)

def ReadLabels(LabelsPathTest):
    # Check if file exists
    if not os.path.isfile(LabelsPathTest):
        print(f"ERROR: Test Labels do not exist in {LabelsPathTest}")
        sys.exit()
    
    # Read CSV file
    df = pd.read_csv(LabelsPathTest)
    
    # Extract the last 8 columns (h1 to h8) as a numpy array
    TestLabels = df.iloc[:, 2: 10].values
    
    return TestLabels

def ConfusionMatrix(LabelsTrue, LabelsPred):
    """
    LabelsTrue - True labels
    LabelsPred - Predicted labels
    """

    # Get the confusion matrix using sklearn.
    LabelsTrue, LabelsPred = list(LabelsTrue), list(LabelsPred)
    cm = confusion_matrix(
        y_true=LabelsTrue, y_pred=LabelsPred  # True class for test-set.
    )  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + " ({0})".format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))

    print("Accuracy: " + str(Accuracy(LabelsPred, LabelsTrue)), "%")


def main():
    """
    Inputs:
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--ModelPath",
        dest="ModelPath",
        # default="../Checkpoints/sup76model.ckpt",
        default="../SavedCheckpoints/9a900model.ckpt",
        help="Path to load latest model from, Default:ModelPath",
    )
    Parser.add_argument(
        "--BasePath",
        dest="BasePath",
        default="../Data/P1Ph2TestSet/",
        help="Path to load images from, Default:BasePath",
    )
    Parser.add_argument(
        "--LabelsPath",
        dest="LabelsPath",
        default="./TxtFiles/LabelsTest.txt",
        help="Path of labels file, Default:./TxtFiles/LabelsTest.txt",
    )
    Parser.add_argument(
        "--ModelType",
        default="UnSup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and UnSup, Default:UnSup",
    )
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    BasePath = Args.BasePath
    LabelsPath = Args.LabelsPath
    ModelType = Args.ModelType

# Generating Testing Data
    input_folder = "../Data/P1Ph2TestSet/Phase2"
    output_folder = "../Data/P1Ph2TestSet/TestSet"
    params = {
        'num_patches': 1,
        'patch_size': (128, 128),
        'rho': 32,
        'edge_margin': 40,
        'resize': (320, 240)

    }
    process_dataset(input_folder, output_folder, **params)
    print("Test Data Generated....")

    # Setup all needed parameters including file reading
    DirNamesTest, ImageSize, NumTestSamples, TestLabels = SetupAll(BasePath)

    TestOperation(BasePath, DirNamesTest, ImageSize, NumTestSamples, TestLabels, ModelType, ModelPath)

    # # Plot Confusion Matrix
    # LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    # ConfusionMatrix(LabelsTrue, LabelsPred)


if __name__ == "__main__":
    main()
