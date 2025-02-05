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
torch.manual_seed(0)

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


def get_homography(H4pt, CA, device='cuda'):

    # Convert inputs to correct format if needed
    if isinstance(H4pt, torch.Tensor):
        H4pt = H4pt.cpu().numpy()
    if isinstance(CA, torch.Tensor):
        CA = CA.cpu().numpy()
    H4pt = H4pt.astype(int)
    # Reshape if necessary
    H4pt = H4pt.reshape(4, 2)
    CA = CA.reshape(4, 2)
    
    # Calculate CB by adding H4pt and CA
    CB = H4pt + CA
    
    # Convert to float32 for cv2
    CA = CA.astype(int)
    CB = CB.astype(int)
    print('CA:', CA)
    print('CB:', CB)
    # Calculate homography using cv2
    H, _ = cv2.findHomography(CA, CB)
    
    # Convert back to torch tensor if needed
    if device == 'cuda':
        H = torch.tensor(H, device=device).float()
    
    return H


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
        H4pt = H4pt.astype(int)
        print('H4pt:', H4pt)

    CA = np.array([[0,0], [0,128], [128,128], [128,0]])

    H_patch2 = get_homography(H4pt, CA, device)
    # Transform to image coordinates
    offset_matrix1 = np.array([[1, 0, x1], [0, 1, y1], [0, 0, 1]])
    offset_matrix2 = np.array([[1, 0, -x2], [0, 1, -y2], [0, 0, 1]])
    
    H_full = offset_matrix1 @ H_patch2 @ offset_matrix2

    
    # Stitch images
    result = stitch_images(img2, img1, H_full)
    return result


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

# Predicted and Ground Truth Corners

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


# Image Stiching
    
    # Load images
    img1 = cv2.imread('2.jpg')
    img2 = cv2.imread('3.jpg')
    img3 = cv2.imread('1.jpg')
    # Perform stitching
    result = main_stitching_pipeline(img1, img2, model, device)
    result1 = main_stitching_pipeline(img3,result,model,device)
    # Save result
    cv2.imwrite('stitched_result_sup_3patches_1.jpg', result)
    cv2.imwrite('stitched_result_sup_3patches.jpg', result1)
    print(result.shape)



if __name__ == "__main__":
    main()
