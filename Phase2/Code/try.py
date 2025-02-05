
import os
import cv2
import numpy as np
import random
import skimage
import PIL
import sys
import pandas as pd
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW



# def ReadLabels(LabelsPathTrain):
#     # Check if file exists
#     if not os.path.isfile(LabelsPathTrain):
#         print(f"ERROR: Train Labels do not exist in {LabelsPathTrain}")
#         sys.exit()
    
#     # Read CSV file
#     df = pd.read_csv(LabelsPathTrain)
    
#     # Extract the last 8 columns (h1 to h8) as a numpy array
#     TrainLabels = df.iloc[:, 2: 10].values
    
#     return TrainLabels


# LabelsPathTrain = '../Data/TrainSet/labels.csv'
# labels = ReadLabels(LabelsPathTrain)

# # Print shape and first few rows for verification
# print("Shape of labels:", labels.shape)
# print("\nFirst few rows:" ,len(labels))
# print(labels[:3])



# # def ReadDirNames(ReadPath):
# #     """
# #     Inputs:
# #     ReadPath is the path of the CSV file you want to read
# #     Outputs:
# #     DirNames is a list of all patch_id values from the CSV file
# #     """
# #     # Check if the file exists
# #     if not os.path.isfile(ReadPath):
# #         print(f"ERROR: File does not exist at {ReadPath}")
# #         return None
    
# #     # Read the CSV file into a DataFrame
# #     df = pd.read_csv(ReadPath)
    
# #     # Extract the 'patch_id' column as a list
# #     DirNames = df['patch_id'].tolist()
    
# #     return DirNames

# # patch_ids = ReadDirNames(LabelsPathTrain)
    
# #     # Print the first few patch IDs for verification
# # if patch_ids:
# #     print("Number of patch IDs:", len(patch_ids))
# #     print("First few patch IDs:", patch_ids[:5])






def find_best_match_and_patches(img1, img2):
    # Convert images to grayscale if they're not already
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1.copy()
        gray2 = img2.copy()
    
    # Use SIFT with increased features
    sift = cv2.SIFT_create(nfeatures=1000)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    # Use FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)
    
    # Sort and get best match
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    best_match = good_matches[0]
    
    # Get patch coordinates
    pt1 = kp1[best_match.queryIdx].pt
    pt2 = kp2[best_match.trainIdx].pt
    
    patch_size = 128
    half_size = patch_size // 2
    
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    
    # Ensure patch coordinates are within bounds
    x1 = max(half_size, min(x1, gray1.shape[1] - half_size))
    y1 = max(half_size, min(y1, gray1.shape[0] - half_size))
    x2 = max(half_size, min(x2, gray2.shape[1] - half_size))
    y2 = max(half_size, min(y2, gray2.shape[0] - half_size))
    
    # Extract patches (single channel)
    patch1 = gray1[y1-half_size:y1+half_size, x1-half_size:x1+half_size]
    patch2 = gray2[y2-half_size:y2+half_size, x2-half_size:x2+half_size]
    
    # Save patches if needed
    cv2.imwrite('patch1.jpg', patch1)
    cv2.imwrite('patch2.jpg', patch2)
    
    return patch1, patch2, (x1, y1), (x2, y2)

def calculate_dlt(H4pt, CA, device='cuda'):
    # Convert numpy arrays to torch tensors on GPU
    H4pt = torch.tensor(H4pt, device=device).float()
    CA = torch.tensor(CA, device=device).float()
    
    # Reshape the input arrays
    H4pt = H4pt.reshape(4, 2)
    CA = CA.reshape(4, 2)
    
    # Initialize matrix A for DLT equations
    A = torch.zeros((8,9), device=device)
    
    # Fill matrix A with equations
    for i in range(4):
        x, y = CA[i]
        u, v = H4pt[i] - CA[i]
        
        A[2*i] = torch.tensor([-x, -y, -1, 0, 0, 0, u*x, u*y, u], device=device)
        A[2*i + 1] = torch.tensor([0, 0, 0, -x, -y, -1, v*x, v*y, v], device=device)
    
    # Solve using SVD
    _, _, Vt = torch.linalg.svd(A)
    
    # Get last row and reshape
    H = Vt[-1].reshape(3, 3)
    
    # Normalize
    H = H / H[2, 2]
    
    return H.cpu().numpy()

def stitch_images(img1, img2, H):
    # Store original images for color conversion later
    orig_img1 = img1.copy()
    orig_img2 = img2.copy()
    
    # Convert to single channel if not already
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create corners matrix
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    
    # Transform corners
    corners1_transformed = cv2.perspectiveTransform(corners1, H)
    
    # Calculate output size
    all_corners = np.vstack((
        corners1_transformed,
        np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    ))
    
    x_min, y_min = np.int32(np.min(all_corners, axis=0)[0] - 0.5)
    x_max, y_max = np.int32(np.max(all_corners, axis=0)[0] + 0.5)
    
    # Translation matrix
    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])
    
    H_final = translation @ H
    
    # Warp and blend
    output_size = (x_max - x_min, y_max - y_min)
    
    # Warp the grayscale image first
    warped_gray = cv2.warpPerspective(img1, H_final, output_size)
    
    # Create binary mask from warped grayscale image
    mask = (warped_gray > 0).astype(np.uint8)
    
    # Warp the color image
    warped_color = cv2.warpPerspective(orig_img1, H_final, output_size)
    
    # Create output canvas (color)
    output = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    
    # Copy second color image
    output[-y_min:h2-y_min, -x_min:w2-x_min] = orig_img2
    
    # Blend using mask
    mask = np.stack([mask] * 3, axis=2)
    output = np.where(mask > 0, warped_color, output)
    
    return output

def main_stitching_pipeline(img1, img2, model, device='cuda'):
    # Get patches
    patch1, patch2, (x1, y1), (x2, y2) = find_best_match_and_patches(img1, img2)
    
    # Stack patches and prepare for model
    stacked_patches = np.stack((patch1, patch2), axis=2)  # Stack along channel dimension
    stacked_patches = torch.from_numpy(stacked_patches).float().to(device)
    stacked_patches = stacked_patches.unsqueeze(0)  # Add batch dimension
    stacked_patches = stacked_patches.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
    stacked_patches = stacked_patches / 255.0  # Normalize
    print(stacked_patches.shape)
    
    # Get H4pt from model
    with torch.no_grad():
        H4pt = model(stacked_patches)
        H4pt = H4pt.cpu().numpy().squeeze()
    
    # Calculate H matrix using DLT
    CA = np.array([[0,0], [0,128], [128,128], [128,0]])
    H_patch = calculate_dlt(H4pt, CA, device)
    
    # Transform to image coordinates
    offset_matrix1 = np.array([[1, 0, x1], [0, 1, y1], [0, 0, 1]])
    offset_matrix2 = np.array([[1, 0, -x2], [0, 1, -y2], [0, 0, 1]])
    
    H_full = offset_matrix2 @ H_patch @ offset_matrix1
    
    # Stitch images
    result = stitch_images(img1, img2, H_full)
    return result

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load your model
    model = HomographyModel().to(device)  # Replace with your model architecture
    #model.load_state_dict(torch.load('../Checkpoints/0a100model.ckpt'))
    model_checkpoint = torch.load('../Checkpoints_UnSup/49model.ckpt')
    model.load_state_dict(model_checkpoint['model_state_dict'])
    #model = model.to(device)
    model.eval()
    
    # Load images
    img1 = cv2.imread('2.jpg')
    img2 = cv2.imread('3.jpg')
    
    # Perform stitching
    result = main_stitching_pipeline(img1, img2, model, device)
    
    # Save result
    cv2.imwrite('stitched_result_unsup.jpg', result)
    print(result.shape)

if __name__ == "__main__":
    main()