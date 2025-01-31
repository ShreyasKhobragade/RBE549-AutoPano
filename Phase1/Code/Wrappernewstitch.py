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
# python load_images.py data/train/set1

import numpy as np
import cv2
from PIL import Image
import math

# Add any python libraries here
import argparse
import os

# Path to the main directory containing image sets

import numpy as np


def ANMS(corners, num_best, original_image):
    # Create a copy of the original image for drawing
    marked_image = original_image.copy()
    
    # Convert corners to a more manageable format
    corners = corners.reshape(-1, 2)  # Convert to Nx2 array
    N_strong = len(corners)
    
    if N_strong == 0:
        return marked_image, []
        
    # Initialize r array
    r = np.full(N_strong, np.inf)
    
    # Calculate r for each point
    for i in range(N_strong):
        x_i, y_i = corners[i]
        
        for j in range(N_strong):
            if i == j:
                continue
                
            x_j, y_j = corners[j]
            
            # Calculate Euclidean distance
            ED = (x_j - x_i)**2 + (y_j - y_i)**2
            
            # Update r[i] if this distance is smaller
            if ED < r[i]:
                r[i] = ED
    
    # Create array of (r, index) pairs for sorting
    corners_with_r = [(r[i], i) for i in range(N_strong)]
    
    # Sort by r in descending order
    corners_with_r.sort(reverse=True)
    
    # Take top num_best points
    selected_indices = [idx for _, idx in corners_with_r[:num_best]]
    selected_corners = corners[selected_indices]
    

    
    # Draw selected corners in green with larger radius
    for corner in selected_corners:
        x, y = np.intp(corner)
        cv2.circle(marked_image, (x, y), 2, (0, 255, 0), -1)  # Green circles

    cv2.putText(marked_image, f'Selected corners: {len(selected_corners)}', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Return corners in the same format as input for compatibility
    selected_corners = selected_corners.reshape(-1, 2)
    
    return marked_image, selected_corners

def features(image, corners):
    # Convert to grayscale if needed
    FD_image=image.copy()
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Reshape corners to Nx2
    corners = corners.reshape(-1, 2)
    
    # Parameters
    patch_size = 41
    output_size = 8
    half_patch = patch_size // 2
    
    # Initialize descriptor array
    num_corners = len(corners)
    descriptors = np.zeros((num_corners, output_size * output_size))
    
    # For each corner point
    for idx, corner in enumerate(corners):
        x, y = np.intp(corner)
        
        # Handle boundary cases
        # Get patch bounds
        y_start = max(y - half_patch, 0)
        y_end = min(y + half_patch + 1, image.shape[0])
        x_start = max(x - half_patch, 0)
        x_end = min(x + half_patch + 1, image.shape[1])
        
        # Extract patch
        patch = np.zeros((patch_size, patch_size))
        actual_patch = image[y_start:y_end, x_start:x_end]
        
        # Calculate where to place the actual patch in the zero-padded patch
        patch_y_start = half_patch - (y - y_start)
        patch_x_start = half_patch - (x - x_start)
        patch[patch_y_start:patch_y_start + actual_patch.shape[0],
              patch_x_start:patch_x_start + actual_patch.shape[1]] = actual_patch
        
        # Apply Gaussian blur
        blurred_patch = cv2.GaussianBlur(patch, (3, 3), 0)
        
        # Subsample to 8x8
        # Calculate indices for evenly spaced samples
        row_indices = np.linspace(0, patch_size-1, output_size, dtype=int)
        col_indices = np.linspace(0, patch_size-1, output_size, dtype=int)
        subsampled = blurred_patch[row_indices][:, col_indices]
        
        
        cv2.circle(FD_image, (x, y), 2, (255, 0, 0), -1)  # Red circles
        cv2.rectangle(FD_image, (x-half_patch,y+half_patch), (x+half_patch,y-half_patch), (255, 255, 0), 1)
        # Reshape to 64x1
        feature_vector = subsampled.reshape(-1)
        
        # Standardize to zero mean and unit variance
        if np.std(feature_vector) != 0:  # Avoid division by zero
            feature_vector = (feature_vector - np.mean(feature_vector)) / np.std(feature_vector)
        
        # Store the feature vector
        descriptors[idx] = feature_vector
        
		
    
    return descriptors, FD_image

def match_features(current_merge,current_image,image_FD,merge_FD,selected_corners_image_corners,selected_corners_merge_corners,match_dir,i):

    
    matched_pairs = []

    img1_features =  merge_FD
    img2_features = image_FD
    img1_corners = selected_corners_merge_corners
    img2_corners = selected_corners_image_corners
    
    keypoints1 = [cv2.KeyPoint(float(x), float(y), 1) for (x, y) in img1_corners]
    keypoints2 = [cv2.KeyPoint(float(x), float(y), 1) for (x,y) in img2_corners]
    
    matches = []
    # For each feature in first image
    for idx1, feat1 in enumerate(img1_features):
        distances = []
        # Compare with all features in second image
        for idx2, feat2 in enumerate(img2_features):
            dist = np.sum((feat1 - feat2) ** 2)  # SSD
            distances.append((dist, idx2))
        
        # Sort distances
        distances.sort(key=lambda x: x[0])
        
        # Apply ratio test
        if len(distances) >= 2:
            ratio = distances[0][0] / distances[1][0]
            if ratio < 0.8:  # Lowe's ratio test threshold
                matches.append((idx1, distances[0][1]))
    
    # Visualize matches
    img1 = current_merge
    img2 = current_image

    # Convert matches to cv2.DMatch format
    good_matches = [cv2.DMatch(_idx1, _idx2, 0) for _idx1, _idx2 in matches]

    # Draw matches
    match_img = cv2.drawMatches(img1, keypoints1, 
                                img2, keypoints2,
                                good_matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                                matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),matchesThickness=1)

    match_filename =f'match_{i}.jpg'
    match_save_path = os.path.join(match_dir, match_filename)
    # Save match visualization
    cv2.imwrite(match_save_path, match_img)
    matched_pairs.append((matches))
    
    return matched_pairs


def compute_homography(points1, points2):
    if len(points1) < 4:
        return None
        
    A = []
    for i in range(len(points1)):
        x, y = points1[i]
        u, v = points2[i]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    
    A = np.array(A)
    
    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    
    # Get the last row of Vt and reshape it to 3x3 matrix
    H = Vt[-1].reshape(3, 3)
    
    # Normalize H
    H = H / H[2, 2]
    
    return H

def apply_homography(H, points):
    """
    Apply homography to points
    """
    # Convert to homogeneous coordinates
    points_homog = np.hstack((points, np.ones((len(points), 1))))
    
    # Apply homography
    transformed_points = np.dot(H, points_homog.T).T
    
    # Convert back to inhomogeneous coordinates
    transformed_points = transformed_points[:, :2] / transformed_points[:, 2:]
    return transformed_points

def match_features_with_ransac(matched_pairs,current_merge,current_image,image_FD,merge_FD,selected_corners_image_corners,selected_corners_merge_corners,match_dir_RANSAC, i, ransac_thresh, max_iters=1000):

    def normalize_points(points):
        """Normalize points for better numerical stability"""
        centroid = np.mean(points, axis=0)
        dist = np.sqrt(np.sum((points - centroid) ** 2, axis=1))
        scale = np.sqrt(2) / np.mean(dist)
        T = np.array([[scale, 0, -scale * centroid[0]],
                     [0, scale, -scale * centroid[1]],
                     [0, 0, 1]])
        points_homog = np.column_stack([points, np.ones(len(points))])
        normalized_points = (T @ points_homog.T).T[:, :2]
        return normalized_points, T

    def compute_homography_normalized(points1, points2):
        """Compute homography with normalized coordinates"""
        # Normalize points
        norm_points1, T1 = normalize_points(points1)
        norm_points2, T2 = normalize_points(points2)
        
        # Compute homography with normalized points
        A = []
        for i in range(len(norm_points1)):
            x, y = norm_points1[i]
            u, v = norm_points2[i]
            A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
            A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
        
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        H_norm = Vt[-1].reshape(3, 3)
        
        # Denormalize
        H = np.linalg.inv(T2) @ H_norm @ T1
        return H / H[2, 2]

    ransac_results = []
    H_matrix = []
    for matches in matched_pairs:
        # if len(matches) < 4:
        #     print(f"Skipping {img1_name}-{img2_name}: Not enough matches ({len(matches)})")
        #     continue

        img1 = current_merge
        img2 = current_image

        # Get corner points
        img1_features =  merge_FD
        img2_features = image_FD
        img1_corners = selected_corners_merge_corners
        img2_corners = selected_corners_image_corners
        
        # Convert matches to point arrays
        points1 = np.float32([img1_corners[m[0]] for m in matches])
        points2 = np.float32([img2_corners[m[1]] for m in matches])
        
        best_inliers = []
        best_H = None
        best_score = 0
        
        # RANSAC loop
        for _ in range(max_iters):
            # Random sample 4 points
            if len(matches) < 4:
                continue
            sample_idx = np.random.choice(len(matches), 4, replace=False)
            sample_points1 = points1[sample_idx]
            sample_points2 = points2[sample_idx]
            
            # Compute homography with normalized coordinates
            try:
                H = compute_homography_normalized(sample_points1, sample_points2)
                
                # Transform all points
                points1_homog = np.column_stack([points1, np.ones(len(points1))])
                transformed_points = (H @ points1_homog.T).T
                denominator = transformed_points[:, 2:]
                denominator[denominator == 0] = 1  # Replace zeros with 1 to avoid division by zero
                transformed_points = transformed_points[:, :2] / denominator
                #transformed_points = transformed_points[:, :2] / transformed_points[:, 2:]
                
                # Compute symmetric transfer error
                errors = np.sqrt(np.sum((transformed_points - points2)**2, axis=1))
                inliers = errors < ransac_thresh
                
                # Compute score (considering both number of inliers and their error)
                score = np.sum(inliers) - np.sum(errors[inliers]) / ransac_thresh
                
                if score > best_score:
                    best_score = score
                    best_inliers = inliers
                    best_H = H
                    
                    # Early termination if we found a very good model
                    inlier_ratio = np.sum(inliers) / len(matches)
                    if inlier_ratio > 0.9:  # 80% inliers
                        break
                        
            except np.linalg.LinAlgError:
                continue
        
        if best_H is not None and len(best_inliers) >= 4:
            # Refine homography using all inliers
            inlier_points1 = points1[best_inliers]
            inlier_points2 = points2[best_inliers]
            refined_H = compute_homography_normalized(inlier_points1, inlier_points2)
            
            # Filter matches using refined model
            filtered_matches = [m for m, is_inlier in zip(matches, best_inliers) if is_inlier]
            
            # Visualize RANSAC matches
            keypoints1 = [cv2.KeyPoint(float(x), float(y), 1) for (x, y) in img1_corners]
            keypoints2 = [cv2.KeyPoint(float(x), float(y), 1) for (x, y) in img2_corners]
            good_matches = [cv2.DMatch(_idx1, _idx2, 0) for _idx1, _idx2 in filtered_matches]
            
            match_img = cv2.drawMatches(
                img1, keypoints1,
                img2, keypoints2,
                good_matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                matchColor=(100, 255, 0),
                singlePointColor=(255, 100, 0),
                matchesThickness=1
            )
            
            # Add text with match statistics
            text = f"Inliers: {len(filtered_matches)}/{len(matches)} ({len(filtered_matches)/len(matches)*100:.1f}%)"
            cv2.putText(match_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            
            match_filename = f'ransac_match_{i}.jpg'
            match_save_path = os.path.join(match_dir_RANSAC, match_filename)
            cv2.imwrite(match_save_path, match_img)
            
            ransac_results.append((filtered_matches, refined_H))
            # print(f"Matched {img1_name}-{img2_name}: {len(filtered_matches)} inliers from {len(matches)} matches")
        H_matrix.append(best_H)
    return ransac_results, H_matrix


def homogeneous_coordinate(point):
    """Ensure division is safe and avoid divide-by-zero errors."""
    epsilon = 1e-10  # Small constant to prevent division by zero
    if abs(point[2][0]) < epsilon:  
        point[2][0] = epsilon  # Avoid divide by zero
    
    return point[0][0] / point[2][0], point[1][0] / point[2][0]  # Normalize

def warp(image, homography):
    print("Warping is started.")

    image_array = np.array(image)
    row_number, column_number = image_array.shape[:2]

    # Compute transformed corner points
    up_left_cor_x, up_left_cor_y = homogeneous_coordinate(np.dot(homography, [[0],[0],[1]]))
    up_right_cor_x, up_right_cor_y = homogeneous_coordinate(np.dot(homography, [[column_number-1],[0],[1]]))
    low_left_cor_x, low_left_cor_y = homogeneous_coordinate(np.dot(homography, [[0],[row_number-1],[1]]))
    low_right_cor_x, low_right_cor_y = homogeneous_coordinate(np.dot(homography, [[column_number-1],[row_number-1],[1]]))

    x_values = [up_left_cor_x, up_right_cor_x, low_right_cor_x, low_left_cor_x]
    y_values = [up_left_cor_y, up_right_cor_y, low_left_cor_y, low_right_cor_y]
    print("x_values: ", x_values, "\n y_values: ", y_values)

    # Compute offsets and image size
    offset_x = math.floor(min(x_values))
    offset_y = math.floor(min(y_values))
    print("offset_x: ", offset_x, "\t offset_y: ", offset_y)

    max_x = math.ceil(max(x_values))
    max_y = math.ceil(max(y_values))

    size_x = max_x - offset_x
    size_y = max_y - offset_y
    print("size_x: ", size_x, "\t size_y: ", size_y)

    # Compute inverse homography
    homography_inverse = np.linalg.inv(homography)
    print("Homography inverse: ", "\n", homography_inverse)

    # Initialize the output image
    result = 255 * np.ones((size_y, size_x, 3), dtype=np.uint8)

    # Warp the image
    for x in range(size_x):
        for y in range(size_y):
            point_xy = homogeneous_coordinate(np.dot(homography_inverse, [[x+offset_x], [y+offset_y], [1]]))
            
            # Ensure valid coordinates
            if np.isinf(point_xy[0]) or np.isnan(point_xy[0]) or np.isinf(point_xy[1]) or np.isnan(point_xy[1]):
                continue  # Skip invalid points
            
            point_x = int(round(point_xy[0]))  # Use rounding to avoid precision errors
            point_y = int(round(point_xy[1]))

            if 0 <= point_x < column_number and 0 <= point_y < row_number:
                result[y, x, :] = image_array[point_y, point_x, :]

    print("Warping is completed.")
    return result, offset_x, offset_y




def blending2images(base_array, image_array, offset_x, offset_y):
    print("Blending two images is started.")

    #image_array = np.array(image_array)
    #base_array = np.array(base_array)

    rows_base, columns_base = int(base_array.shape[0]), int(base_array.shape[1])
    rows_image, columns_image = int(image_array.shape[0]), int(image_array.shape[1])

    print("Column number of base: ", columns_base, "\t Row number of base: ", rows_base)
    print("Column number of image: ", columns_image, "\t Row number of image: ", rows_image)

    x_min = 0
    if offset_x>0:
        x_max = max([offset_x+columns_image, columns_base])
    else:
        x_max = max([-offset_x + columns_base, columns_image])

    y_min = 0
    # note that offset_y was always negative in this assignment.
    y_max = max([rows_base-offset_y, rows_image])

    size_x = x_max - x_min
    size_y = y_max - y_min

    print("size_x: ", size_x, "\t size_y: ", size_y)
    blending = 255*np.ones((size_y, size_x, 3))

    # right to left image stitching
    if offset_x > 0:
        blending[:rows_image, offset_x:columns_image+offset_x, :] = image_array[:, :, :]
        blending[-offset_y:rows_base-offset_y, :columns_base, :] = base_array[:, :, :]
    # left to right image stitching
    else:
        blending[:rows_image, :columns_image, :] = image_array[:, :, :]
        blending[-offset_y:rows_base-offset_y, -offset_x:columns_base-offset_x, :] = base_array[:, :, :]

    print("Blending is completed.")
    return blending



def stitch_images(image1, image2, homography_matrix):

    # Compute the size of the warped image
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    # Compute the corners of image2 after applying the homography
    corners = np.array([
        [0, 0],
        [width2, 0],
        [width2, height2],
        [0, height2]
    ], dtype=np.float32)
    transformed_corners = cv2.perspectiveTransform(np.array([corners]), homography_matrix)[0]

    # Determine the size of the output canvas
    min_x = min(0, transformed_corners[:, 0].min())
    min_y = min(0, transformed_corners[:, 1].min())
    max_x = max(width1, transformed_corners[:, 0].max())
    max_y = max(height1, transformed_corners[:, 1].max())

    # Offset for negative coordinates
    offset_x = int(abs(min_x)) if min_x < 0 else 0
    offset_y = int(abs(min_y)) if min_y < 0 else 0

    # Compute the size of the final stitched image
    output_width = int(max_x - min_x)
    output_height = int(max_y - min_y)

    # Adjust the homography to include the offset
    translation_matrix = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ])
    adjusted_homography = np.dot(translation_matrix, homography_matrix)

    # Warp the second image onto the canvas
    warped_image2 = cv2.warpPerspective(image2, adjusted_homography, (output_width, output_height))
    
    # Place the first image onto the canvas
    stitched_image = warped_image2.copy()
    stitched_image[offset_y:offset_y + height1, offset_x:offset_x + width1] = image1

    # Blend overlapping regions
    for y in range(offset_y, offset_y + height1):
        for x in range(offset_x, offset_x + width1):
            if np.any(warped_image2[y, x] > 0) and np.any(image1[y - offset_y, x - offset_x] > 0):
                # Average overlapping pixels
                stitched_image[y, x] = (
                    0.5 * warped_image2[y, x] + 0.5 * image1[y - offset_y, x - offset_x]
                ).astype(np.uint8)

    return stitched_image


def warp_two_images(image1, image2, H, blending=False, poisson=False):


    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    p1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    p2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    p2_ = cv2.perspectiveTransform(p2, H)
    p = np.concatenate((p1, p2_), axis=0)

    [xmin, ymin] = np.int32(p.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(p.max(axis=0).ravel() + 0.5)

    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate

    result = cv2.warpPerspective(image2, Ht.dot(H), (xmax - xmin, ymax - ymin))

    if blending:
        if poisson:
            mask = 255 * np.ones_like(image1, dtype=np.uint8)

            # Find the overlapping region of the two images and define the mask
            # cv2.imwrite(os.path.join(save_path, "mask.png"), mask)

            # Perform Poisson blending
            result = cv2.seamlessClone(image1, result, mask, (int(t[0] + w1/2), int(t[1] + h1/2)), cv2.NORMAL_CLONE)

    else:
        result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = image1

    return result


def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--path", type=str, help="Path to the main directory containing image sets")
    Parser.add_argument("--Threshold", type=float, help="RANSAC Threshold")
    Parser.add_argument("--Nmax", type=int, help="RANSAC iterations")

    Args = Parser.parse_args()

    # Get the directory path from the command line
    main_directory = os.path.abspath(Args.path)
    set_name = os.path.basename(main_directory)

    # Define the output directory for corner-detected images
    corner_dir = os.path.join(os.path.dirname(os.path.dirname(main_directory)), "corner", set_name)

    # Create the output folder if it doesn't exist
    os.makedirs(corner_dir, exist_ok=True)
    corner_dir_ANMS = os.path.join(os.path.dirname(os.path.dirname(main_directory)), "corner_ANMS", set_name)

    match_dir = os.path.join(os.path.dirname(os.path.dirname(main_directory)), "match", set_name)
    os.makedirs(match_dir, exist_ok=True)
    match_dir_RANSAC = os.path.join(os.path.dirname(os.path.dirname(main_directory)), "ransac_match", set_name)
    os.makedirs(match_dir_RANSAC, exist_ok=True)
    pano_dir = os.path.join(os.path.dirname(os.path.dirname(main_directory)), "Pano", set_name)
    os.makedirs(pano_dir, exist_ok=True)

    # Create the output folder if it doesn't exist
    os.makedirs(corner_dir_ANMS, exist_ok=True)

    Feature = os.path.join(os.path.dirname(os.path.dirname(main_directory)), "Feature", set_name)

    # Create the output folder if it doesn't exist
    os.makedirs(Feature, exist_ok=True)

    # List to store images
    images = []
    image_corners=[]
    ANMS_corners = []
    ANMS_Images = []
    Features_D=[]
    image_data = {}

# Read images from the folder and store them in a list images
    print(f"Reading images from {main_directory}")
    if not os.path.exists(main_directory):
        print(f"Error: Directory '{main_directory}' not found.")
        exit(1)    # Iterate over each image in the folder
    for image_name in sorted(os.listdir(main_directory)):
        image_path = os.path.join(main_directory, image_name)
        # Read image using OpenCV
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)  # Append image to list
            print(f"Loaded image[{len(images) - 1}] from {image_path}")
        else:
            print(f"Could not read {image_path}")
        
    # images = images.reverse()

    # images variable has list of all images

    for i in range(len(images)-1):
        if i == 0:
            # current_merge, _, _ = ProjectOntoCylinder(images[i])
            current_merge = images[i]



        current_image = images[i+1]

        current_image_p= cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        current_merge_p= cv2.cvtColor(current_merge, cv2.COLOR_BGR2GRAY)

        # Apply Shi-Tomasi Corner Detection
        image_corners = cv2.goodFeaturesToTrack(current_image_p, maxCorners=2000, qualityLevel=0.01, minDistance=8)
        merge_corners = cv2.goodFeaturesToTrack(current_merge_p, maxCorners=2000, qualityLevel=0.01, minDistance=8)

        if i == 0:
            corner_image = images[i].copy()
            if merge_corners is not None:
                merge_corners = np.intp(merge_corners)
                for corner in merge_corners:
                    x, y = corner.ravel()
                    cv2.circle(corner_image, (x, y), 2, (0, 0, 255), -1)  # Mark in Green (BGR: (0, 255, 0))
            save_path = corner_dir + '/corner_' + str(i+1) + '.png'
            cv2.imwrite(save_path, corner_image)

            

        corner_image = images[i+1].copy()
        
        if image_corners is not None:
            image_corners = np.intp(image_corners)  # Convert to integer values        
            for corner in image_corners:
                x, y = corner.ravel()
                cv2.circle(corner_image, (x, y), 2, (0, 0, 255), -1)  # Mark in Green (BGR: (0, 255, 0))
        save_path = corner_dir + '/corner_' + str(i+2) + '.png'
        cv2.imwrite(save_path, corner_image)

        # Perform ANMS


        num_best = 800
        marked_image_corners, selected_corners_image_corners = ANMS(image_corners, num_best, current_image)
        marked_merge_corners, selected_corners_merge_corners = ANMS(merge_corners, num_best, current_merge)

        if i == 0:
            save_path_merge_corner = os.path.join(corner_dir_ANMS, 'ANMScorner_' + str(i+1) + '.png')
            cv2.imwrite(save_path_merge_corner, marked_merge_corners)
            
        save_path_image_corner = os.path.join(corner_dir_ANMS, 'ANMScorner_' + str(i+2) + '.png')
        save_path_merge_corner = os.path.join(corner_dir_ANMS, 'ANMScorner_merge_' + str(i+1) + '.png')
        cv2.imwrite(save_path_image_corner, marked_image_corners)
        cv2.imwrite(save_path_merge_corner, marked_merge_corners)


            
        image_FD, FD_image_current=features(current_image, selected_corners_image_corners)
        merge_FD, FD_image_merge=features(current_merge, selected_corners_merge_corners)

        if i == 0:
            save_path = os.path.join(Feature, 'Feature_' + str(i+1) + '.png')
            cv2.imwrite(save_path, FD_image_merge)

        save_path = os.path.join(Feature,'Feature_' + str(i+2) + '.png')
        cv2.imwrite(save_path, FD_image_current)
        save_path = os.path.join(Feature, 'Feature_merge' + str(i+1) + '.png')
        cv2.imwrite(save_path,FD_image_merge)
    


        print('loop :', i)
        matched_pairs = match_features(current_merge,current_image,image_FD,merge_FD,selected_corners_image_corners,selected_corners_merge_corners,match_dir,i)
        match_pairs_ransac, H = match_features_with_ransac(matched_pairs,current_merge,current_image,image_FD,merge_FD,selected_corners_image_corners,selected_corners_merge_corners,match_dir_RANSAC, i,ransac_thresh=10.0, max_iters=1000)
        # match_pairs_ransac, H = match_features_with_ransac(matched_pairs,current_merge,current_image,image_FD,merge_FD,selected_corners_image_corners,selected_corners_merge_corners,match_dir_RANSAC, i,ransac_thresh=15.0, max_iters=1000)
        print('Homography:', H)

        # H[0][2] = H[0][2]*[0, 0, 1]
        # StitchedImage = StitchImages(current_image, current_merge, H[0])
        # warped_image_source1, source1_offset_x, source1_offset_y = warp(current_merge, H[0])
        # #warped_image_source1, source1_offset_x, source1_offset_y = warp(current_image, np.linalg.inv(H[0]))
        # image1 = Image.fromarray(warped_image_source1.astype('uint8'), 'RGB')
        # image1.save("warped_image.jpg")
        # #result = cv2.seamlessClone(current_image, current_merge, mask, center, cv2.MIXED_CLONE)
        # result_blended = blending2images(current_image, warped_image_source1, source1_offset_x, source1_offset_y)
        # cv2.imshow("Blended Image", result_blended)
        # image_final = Image.fromarray(result_blended.astype('uint8'), 'RGB')
        # image_final.save("blended_image.jpg")
        # print("Blended image is generated.")
        # #print(result_blended.shape)

        # warped_image_source1, source1_offset_x, source1_offset_y = warp(current_merge, H[0])
        # # warped_image_source1, source1_offset_x, source1_offset_y = warp(current_image, np.linalg.inv(H[0]))
        # image1 = Image.fromarray(warped_image_source1.astype('uint8'), 'RGB')
        # image1.save("warped_image.jpg")
        # #result = cv2.seamlessClone(current_image, current_merge, mask, center, cv2.MIXED_CLONE)
        # result_blended = blending2images(current_image, warped_image_source1, source1_offset_x, source1_offset_y)
        # # result_blended = blending2images(current_merge, warped_image_source1, source1_offset_x, source1_offset_y)

        # cv2.imshow("Blended Image", result_blended)
        # image_final = Image.fromarray(result_blended.astype('uint8'), 'RGB')
        # image_final.save("blended_image.jpg")
        # print("Blended image is generated.")
        
        if i<len(images)//2:
        # if i%2 ==0:
            result = stitch_images(current_image, current_merge, H[0])
            # result = stitch_images1(current_merge, current_image, np.linalg.inv(H[0]))
            # result = warp_two_images(current_merge, current_image, np.linalg.inv(H[0]))

            

            # cv2.imwrite("stitched_result.jpg", result)
        else:
            result = stitch_images(current_merge, current_image, np.linalg.inv(H[0]))
            # result = warp_two_images(current_image, current_merge, H[0])
            # result = stitch_images1(current_image, current_merge, H[0])


            # cv2.imwrite("stitched_result.jpg", result)
        
        cv2.imwrite(f"stitched_image_{i+1}.jpg", result)

        # current_merge = StitchedImage.copy()
        current_merge = np.uint8(result)
        # cv2.imwrite('merge_image_new_method.png',current_merge)


if __name__ == "__main__":
    main()
	
"""
Read a set of images for Panorama stitching
"""

"""
Corner Detection
Save Corner detection output as corners.png
"""

"""
Perform ANMS: Adaptive Non-Maximal Suppression
Save ANMS output as anms.png
"""

"""
Feature Descriptors
Save Feature Descriptor output as FD.png
"""

"""
Feature Matching
Save Feature Matching output as matching.png
"""

"""
Refine: RANSAC, Estimate Homography
"""

"""
Image Warping + Blending
Save Panorama output as mypano.png
"""