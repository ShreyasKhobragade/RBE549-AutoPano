#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Code starts here:

import numpy as np
import cv2
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from Network.Network import HomographyModel, LossFn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add any python libraries here

def generate_patches(image, num_patches=1, patch_size=(128, 128), rho=32, edge_margin=40, max_attempts=40, resize=(320, 240)):
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
            perturb += np.random.randint(0,8, size=(1, 2))
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


def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """
    
    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""
    # Load your model
    model = HomographyModel().to(device)
    model_checkpoint = torch.load('../Checkpoints/199model.ckpt')
    model.load_state_dict(model_checkpoint['model_state_dict'])
    model.eval()

    # Load image and generate patches
    image_path = '../Data/Train/1.jpg'
    image = cv2.imread(image_path)
    image = cv2.resize(image, (320, 240))
    patches, labels, corners_list = generate_patches(image)

    # Compute ground truth corners
    ground_truth = labels[0] + corners_list[0]
    ground_truth = ground_truth.reshape(4, 2).astype(int)

    # Prepare input for model
    stacked_patches = patches[0]
    stacked_patches = torch.from_numpy(stacked_patches).float().to(device)
    stacked_patches = stacked_patches.unsqueeze(0).permute(0, 3, 1, 2).contiguous() / 255.0

    # Predict H4pt from model
    with torch.no_grad():
        H4pt = model(stacked_patches)
        H4pt = H4pt.cpu().numpy().squeeze()

    # Compute predicted corners
    predicted = H4pt + corners_list[0]
    predicted = predicted.reshape(4, 2).astype(int)

    # Interchange the 3rd and 4th points in ground_truth
    ground_truth[[2, 3]] = ground_truth[[3, 2]]

    # Interchange the 3rd and 4th points in predicted
    predicted[[2, 3]] = predicted[[3, 2]]


    # Draw ground truth polygon on the image
    ground_truth_polygon = ground_truth[:, np.newaxis, :]  # Reshape to (n_points, 1, 2)
    cv2.polylines(image, [ground_truth_polygon], isClosed=True, color=(255, 0, 0), thickness=1)

    # Draw predicted polygon on the image
    predicted_polygon = predicted[:, np.newaxis, :]       # Reshape to (n_points, 1, 2)
    cv2.polylines(image, [predicted_polygon], isClosed=True, color=(0, 255, 0), thickness=1)

    # Display the image with polygons
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
