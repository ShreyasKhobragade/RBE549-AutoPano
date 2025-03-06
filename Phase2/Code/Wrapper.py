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


# Code starts here:

import numpy as np
import cv2
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from Network.Network import HomographyModel, LossFn
import matplotlib.pyplot as plt
import argparse
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.manual_seed(0)
# np.random.seed(43)

# Add any python libraries here

def epe(predicted, ground_truth):
    return np.mean(np.sqrt(np.sum((predicted - ground_truth)**2, axis=1)))

def ANMS(corners, num_best, original_image):

    
    corners = corners.reshape(-1, 2)  
    Num_Strong = len(corners)
            
    r = np.full(Num_Strong, np.inf)
    
    for i in range(Num_Strong):
        x_i, y_i = corners[i]
        
        for j in range(Num_Strong):
            if i == j:
                continue
                
            x_j, y_j = corners[j]
            
            Dist = (x_j - x_i)**2 + (y_j - y_i)**2
            
            if Dist < r[i]:
                r[i] = Dist
    
    corners_with_r = [(r[i], i) for i in range(Num_Strong)]
    
    corners_with_r.sort(reverse=True)
    
    selected_indices = [idx for _, idx in corners_with_r[:num_best]]
    selected_corners = corners[selected_indices]   

    selected_corners = selected_corners.reshape(-1, 2)
    
    return selected_corners

def features(image, corners):

    FD_image=image.copy()
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners = corners.reshape(-1, 2)
    
    patch_size = 41
    output_size = 8
    half_patch = patch_size // 2
    
    num_corners = len(corners)
    descriptors = np.zeros((num_corners, output_size * output_size))
    
    for id, corner in enumerate(corners):
        x, y = np.intp(corner)
        
        y_start = max(y - half_patch, 0)
        y_end = min(y + half_patch + 1, image.shape[0])
        x_start = max(x - half_patch, 0)
        x_end = min(x + half_patch + 1, image.shape[1])
        
        patch = np.zeros((patch_size, patch_size))
        actual_patch = image[y_start:y_end, x_start:x_end]
        
        patch_y_start = half_patch - (y - y_start)
        patch_x_start = half_patch - (x - x_start)
        patch[patch_y_start:patch_y_start + actual_patch.shape[0],
              patch_x_start:patch_x_start + actual_patch.shape[1]] = actual_patch
        
        blurred_patch = cv2.GaussianBlur(patch, (3, 3), 0)
        
        row_indices = np.linspace(0, patch_size-1, output_size, dtype=int)
        col_indices = np.linspace(0, patch_size-1, output_size, dtype=int)
        subsampled = blurred_patch[row_indices][:, col_indices]
        
        
        cv2.circle(FD_image, (x, y), 2, (255, 0, 0), -1)  
        cv2.rectangle(FD_image, (x-half_patch,y+half_patch), (x+half_patch,y-half_patch), (255, 255, 0), 1)
        feature_vector = subsampled.reshape(-1)
        
        if np.std(feature_vector) != 0:  
            feature_vector = (feature_vector - np.mean(feature_vector)) / np.std(feature_vector)
        
        descriptors[id] = feature_vector
           
    return descriptors, FD_image

def match_features(current_merge,current_image,image_FD,merge_FD,selected_corners_image_corners,selected_corners_merge_corners):
    matched_pairs = []
    img1_features =  merge_FD
    img2_features = image_FD    
    matches = []
    for id1, feat1 in enumerate(img1_features):
        distances = []
        for id2, feat2 in enumerate(img2_features):
            dist = np.sum((feat1 - feat2) ** 2)  # SSD
            distances.append((dist, id2))
        
        distances.sort(key=lambda x: x[0])
        
        if len(distances) >= 2:
            ratio = distances[0][0] / distances[1][0]
            if ratio < 0.7:  # Lowe's ratio test threshold
                matches.append((id1, distances[0][1]))
    
        matched_pairs.append((matches))
    
    return matched_pairs

def compute_homography(src_pts, dst_pts):
    assert len(src_pts) == len(dst_pts) and len(src_pts) >= 4
    A = []
    for (x, y), (x_prime, y_prime) in zip(src_pts, dst_pts):
        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
    
    A = np.array(A)
    
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    
    H = h.reshape(3, 3)
    
    H = H / H[2, 2]
    return H

def ransac_homography(matched_pairs, img1_corners, img2_corners, current_merge, current_image, 
                      threshold=5.0, max_iterations=1000, edge_threshold=10):
    matches = matched_pairs[0]
    best_inliers = []
    max_inliers = 0
    best_H = None
    
    src_pts = np.float32(img1_corners)
    dst_pts = np.float32(img2_corners)
    
    valid_matches = []
    for idx, match in enumerate(matches):            
        valid_matches.append(match)
    
    if len(valid_matches) < 4:
        return [], None
    
    for _ in range(max_iterations):
        random_indices = np.random.choice(len(valid_matches), 4, replace=False)
        sample_matches = [valid_matches[idx] for idx in random_indices]
        
        src_sample = np.float32([src_pts[match[0]] for match in sample_matches])
        dst_sample = np.float32([dst_pts[match[1]] for match in sample_matches])
        
        H = compute_homography(src_sample, dst_sample)
        
        if H is None:
            continue
        
        inliers = []
        for idx, match in enumerate(valid_matches):
            src = src_pts[match[0]]
            dst = dst_pts[match[1]]
            
            src_homogeneous = np.array([src[0], src[1], 1])
            dst_projected = H @ src_homogeneous
            dst_projected = dst_projected / dst_projected[2]
            
            error = np.sqrt(np.sum((dst - dst_projected[:2]) ** 2))
            
            if error < threshold:
                inliers.append(idx)
        
        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_inliers = inliers
            best_H = H
    
    return best_inliers, best_H

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
    # print('CA:', CA)
    # print('CB:', CB)
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
    # print(stacked_patches.shape)
    
    # Get H4pt from model
    with torch.no_grad():
        H4pt = model(stacked_patches)
        H4pt = H4pt.cpu().numpy().squeeze()
        # print('H4pt:', H4pt)
        # H4pt = 8 * H4pt
        H4pt = H4pt.astype(int)
        # print('H4pt:', H4pt)
        


    CA = np.array([[0,0], [0,128], [128,128], [128,0]])

    H_patch2 = get_homography(H4pt, CA, device)
    # Transform to image coordinates
    offset_matrix1 = np.array([[1, 0, x1], [0, 1, y1], [0, 0, 1]])
    offset_matrix2 = np.array([[1, 0, -x2], [0, 1, -y2], [0, 0, 1]])
    
    H_full = offset_matrix1 @ H_patch2 @ offset_matrix2

    
    # Stitch images
    result = stitch_images(img2, img1, H_full)
    return result

def classical_method(stacked_patches, CA):

    cv2.imwrite('patch1.jpg', stacked_patches[:, :, 0])
    cv2.imwrite('patch2.jpg', stacked_patches[:, :, 1])
    patcha = stacked_patches[:, :, 0]
    patchb = stacked_patches[:, :, 1]
    pa = cv2.goodFeaturesToTrack(patcha, maxCorners=200, qualityLevel=0.01, minDistance=1)
    pb = cv2.goodFeaturesToTrack(patchb, maxCorners=200, qualityLevel=0.01, minDistance=1)
    num_best = 80
    selected_corners_image_corners = ANMS(pa, num_best, patchb)
    selected_corners_merge_corners = ANMS(pb, num_best, patchb)
    image_FD, _ =features(patcha, selected_corners_image_corners)
    merge_FD, _ =features(patchb, selected_corners_merge_corners)

    matched_pairs = match_features(patchb, patcha,image_FD,merge_FD,selected_corners_image_corners,selected_corners_merge_corners)
    _, H = ransac_homography(matched_pairs, selected_corners_merge_corners, selected_corners_image_corners, patchb, patcha,  threshold=5.0, max_iterations=10, edge_threshold=1) 
    # print(H)

    CA = CA.reshape(-1, 1, 2).astype(np.float32)
    # H = np.linalg.inv(H)
    classical_corners = cv2.perspectiveTransform(CA, H)
    return classical_corners


def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--path", type=str, default="../Data/P1Ph2TestSet/Phase2Pano/tower", help="Path to the main directory containing image sets")
    Parser.add_argument("--ModelType", type=str,default="Sup", help="Sup or UnSup for Stitching")
    Parser.add_argument("--CheckPointPath",  default="../Checkpoints/", help="Path to save Checkpoints, Default: ../Checkpoints/",
    )
    Args = Parser.parse_args()

    CheckPointPath = Args.CheckPointPath

    main_directory_1=os.path.abspath(Args.path)
    subfolder_name = os.path.basename(main_directory_1)
    set_name = os.path.basename(main_directory_1)
    results = os.path.join(os.path.dirname(os.path.dirname(main_directory_1)),'Results',subfolder_name)
    os.makedirs(results, exist_ok=True)
    #results_directory = os.path.join('Data', 'TestResults', subfolder_name)

    # torch.cuda.empty_cache()

    # Load your Supervised model
    models = HomographyModel().to(device)
    models_checkpoint = torch.load(CheckPointPath)
    models.load_state_dict(models_checkpoint['model_state_dict'])
    models.eval()

    # Unsupervised Model
    modelu = HomographyModel().to(device)
    modelu_checkpoint = torch.load(CheckPointPath)
    modelu.load_state_dict(modelu_checkpoint['model_state_dict'])
    modelu.eval()



# # Predicted and Ground Truth Corners

#     # Load image and generate patches
#     image_path = '../Data/Train/3701.jpg'
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (320, 240))
#     patches, labels, corners_list = generate_patches(image)

#     # Compute ground truth corners
#     ground_truth = labels[0] + corners_list[0]
#     ground_truth = ground_truth.reshape(4, 2).astype(int)

#     # Classical Method
#     classical_corners = classical_method(patches[0], corners_list[0])
#     classical_corners = classical_corners.astype(int).reshape(-1, 2)

#     # Prepare input for model
#     stacked_patches = patches[0]
#     stacked_patches = torch.from_numpy(stacked_patches).float().to(device)
#     stacked_patches = stacked_patches.unsqueeze(0).permute(0, 3, 1, 2).contiguous() / 255.0

#     # Predict H4pt from model
#     with torch.no_grad():
#         H4pts = models(stacked_patches)
#         H4pts = H4pts.cpu().numpy().squeeze()
#         H4ptu = modelu(stacked_patches)
#         H4ptu = H4ptu.cpu().numpy().squeeze()

#     # Compute Supervised predicted corners
#     predicteds = H4pts + corners_list[0]
#     predicteds = predicteds.reshape(4, 2).astype(int)
#     # Compute Unsupervised predicted corners
#     predictedu = H4ptu + corners_list[0]
#     predictedu = predictedu.reshape(4, 2).astype(int)

#     # Interchange the 3rd and 4th points in ground_truth
#     ground_truth[[2, 3]] = ground_truth[[3, 2]]

#     # Interchange the 3rd and 4th points in predicted
#     predicteds[[2, 3]] = predicteds[[3, 2]]
#     predictedu[[2, 3]] = predictedu[[3, 2]]

#     # Interchange the 3rd and 4th points in classical
#     classical_corners[[2, 3]] = classical_corners[[3, 2]]


#     # Draw ground truth polygon on the image
#     ground_truth_polygon = ground_truth[:, np.newaxis, :]  # Reshape to (n_points, 1, 2)
#     cv2.polylines(image, [ground_truth_polygon], isClosed=True, color=(0, 0, 255), thickness=1)

#     # Draw supervised predicted polygon on the image
#     predicteds_polygon = predicteds[:, np.newaxis, :]       # Reshape to (n_points, 1, 2)
#     cv2.polylines(image, [predicteds_polygon], isClosed=True, color=(0, 255, 0), thickness=1)

#     # Draw unsupervised predicted polygon on the image
#     predictedu_polygon = predictedu[:, np.newaxis, :]       # Reshape to (n_points, 1, 2)
#     cv2.polylines(image, [predictedu_polygon], isClosed=True, color=(255, 0, 0), thickness=1)

#     classical_polygon = classical_corners[:, np.newaxis, :]      # Reshape to (n_points, 1, 2)
#     cv2.polylines(image, [classical_polygon], isClosed=True, color=(0, 255, 255), thickness=1)

#     epe_sup = epe(predicteds, ground_truth)
#     epe_unsup = epe(predictedu, ground_truth)
#     epe_classical = epe(classical_corners, ground_truth)
#     print(f"Supervised EPE: {epe_sup}")
#     print(f"Unsupervised EPE: {epe_unsup}")
#     print(f"Classical EPE: {epe_classical}")

#     cv2.imwrite('overlap.jpg', image)
#     cv2.waitKey(500)    


# # Image Stiching

    if Args.ModelType == 'Sup':
        model = models
    if Args.ModelType == 'UnSup':
        model = modelu
    else:
        model = models

    images = []

    if not os.path.exists(main_directory_1):
        print(f"Error: Directory '{main_directory_1}' not found.")
        exit(1)   
    for image_name in sorted(os.listdir(main_directory_1)):
        image_path = os.path.join(main_directory_1, image_name)
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)  # Append image to list
            print(f"Loaded image[{len(images) - 1}] from {image_path}")
        else:
            print(f"Could not read {image_path}")

    result = images[0]  # Start with first image
    for i in range(1, len(images)):
        # Stitch current result with next image
        if i ==1:
            result = main_stitching_pipeline(result, images[i], model, device)
        
        else: 
            result = main_stitching_pipeline(images[i], result, model, device)
        # Save intermediate result
        output_path = os.path.join(results, f'mypano{i}.jpg')
        cv2.imwrite(output_path, result)
        print(f"Saved stitched result {i} to {output_path}")
 
if __name__ == "__main__":
    main()
