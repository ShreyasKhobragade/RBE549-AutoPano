"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import kornia  # You can use this to get the transform and warp in this project
import pytorch_lightning as pl

# Don't generate pyc codes
sys.dont_write_bytecode = True


# def LossFn(delta, img_a, patch_b, corners):
#     ###############################################
#     # Fill your loss function of choice here!
#     ###############################################

#     ###############################################
#     # You can use kornia to get the transform and warp in this project
#     # Bonus if you implement it yourself
#     ###############################################
#     loss_fn=nn.CrossEntropyLoss()
#     loss = ...
#     return loss

# def LossFn(PredicatedCoordinatesBatch, CoordinatesBatch):
#     # Reshape delta from (batch_size, 168) → (batch_size, 8, 21)
#     loss_fn=torch.nn.MSELoss()
#     loss=loss_fn(PredicatedCoordinatesBatch, CoordinatesBatch)

#     return loss

def LossFn(PredicatedCoordinatesBatch, CoordinatesBatch):
    # Ensure both inputs are float32
    PredicatedCoordinatesBatch = PredicatedCoordinatesBatch.float()
    CoordinatesBatch = CoordinatesBatch.float()

    # Compute loss using MSELoss
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(PredicatedCoordinatesBatch, CoordinatesBatch)

    return torch.sqrt(loss)



class HomographyModel(pl.LightningModule):
    def __init__(self):
        super(HomographyModel, self).__init__()
        # self.hparams = hparams
        self.model = SupNet()

    def forward(self, a):
        return self.model(a)

    def training_step(self, batch, batch_idx):
        img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(patch_a, patch_b)
        loss = LossFn(delta, img_a, patch_b, corners)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, labels):
        prediction = self.model(batch)
        loss = LossFn(prediction, labels)
        # img_a, patch_a, patch_b, corners, gt = batch
        # delta = self.model(patch_a, patch_b)
        # loss = LossFn(delta, img_a, patch_b, corners)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


# class Net(nn.Module):
#     def __init__(self, InputSize, OutputSize):
#         """
#         Inputs:
#         InputSize - Size of the Input
#         OutputSize - Size of the Output
#         """
#         super().__init__()
#         #############################
#         # Fill your network initialization of choice here!
#         #############################
#         ...
#         #############################
#         # You will need to change the input size and output
#         # size for your Spatial transformer network layer!
#         #############################
#         # Spatial transformer localization-network
#         self.localization = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(8, 10, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#         )

#         # Regressor for the 3 * 2 affine matrix
#         self.fc_loc = nn.Sequential(
#             nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
#         )

#         # Initialize the weights/bias with identity transformation
#         self.fc_loc[2].weight.data.zero_()
#         self.fc_loc[2].bias.data.copy_(
#             torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
#         )

#     #############################
#     # You will need to change the input size and output
#     # size for your Spatial transformer network layer!
#     #############################
#     def stn(self, x):
#         "Spatial transformer network forward function"
#         xs = self.localization(x)
#         xs = xs.view(-1, 10 * 3 * 3)
#         theta = self.fc_loc(xs)
#         theta = theta.view(-1, 2, 3)

#         grid = F.affine_grid(theta, x.size())
#         x = F.grid_sample(x, grid)

#         return x

#     def forward(self, xa, xb):
#         """
#         Input:
#         xa is a MiniBatch of the image a
#         xb is a MiniBatch of the image b
#         Outputs:
#         out - output of the network
#         """
#         #############################
#         # Fill your network structure of choice here!
#         #############################
#         return out

class SupNet(nn.Module):
    def __init__(self, InputSize=2, OutputSize=8):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output (should be 168 for 21 bins * 8 dimensions)
        """

        super().__init__()
        
        # VGG-style convolutional blocks for main network
        self.conv1_1 = nn.Conv2d(InputSize, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        
        self.conv4_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(128)
        self.conv4_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(128)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers for main network
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, OutputSize)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )
        
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    # def stn(self, x):
    #     """Spatial transformer network forward function"""
    #     xs = self.localization(x)
    #     xs = xs.view(-1, 10 * 3 * 3)
    #     theta = self.fc_loc(xs)
    #     theta = theta.view(-1, 2, 3)
    #     grid = F.affine_grid(theta, x.size())
    #     x = F.grid_sample(x, grid)
    #     return x

    def forward(self, x):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network (168 dimensional for 21 bins * 8 corners)
        """

        #x = torch.cat([xa, xb], dim=1)
        
        # First conv block
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool(x)
        
        # Second conv block
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool(x)
        
        # Third conv block
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool(x)
        
        # Fourth conv block
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool(x)
        
        # Flatten and apply dropout
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        
        # Fully connected layers with dropout
        #x = F.relu(self.fc1(x))
        x = self.fc1(x)
        out = self.fc2(x)
        
        # Reshape to (batch_size, 8, 21) and apply softmax
        # x = x.view(-1, 8, 21)
        # x = F.softmax(x, dim=2)
        
        # # Reshape back to (batch_size, 168)
        # out = x.view(-1, 168)
        
        return out