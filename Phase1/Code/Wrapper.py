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
# python Wrapper.py --path ../Data/Train/Set1 --Threshold 5.0 --Nmax 1000 --EdgeThresh 10 

import numpy as np
import cv2
import math
import argparse
import os

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
        
    # Add text showing number of corners
    # cv2.putText(marked_image, f'Total corners: {N_strong}', (10, 30), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #cv2.putText(marked_image, f'Selected corners: {len(selected_corners)}', (10, 60), 
                #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
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
            if ratio < 0.7:  # Lowe's ratio test threshold
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

    match_filename =f'matching_{i}.jpg'
    match_save_path = os.path.join(match_dir, match_filename)
    # Save match visualization
    cv2.imwrite(match_save_path, match_img)
    matched_pairs.append((matches))
    
    return matched_pairs


def compute_homography1(points1, points2):

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

    # Convert to homogeneous coordinates
    points_homog = np.hstack((points, np.ones((len(points), 1))))
    
    # Apply homography
    transformed_points = np.dot(H, points_homog.T).T
    
    # Convert back to inhomogeneous coordinates
    transformed_points = transformed_points[:, :2] / transformed_points[:, 2:]
    return transformed_points

def compute_homography(src_pts, dst_pts):
    """
    Compute homography matrix using Direct Linear Transformation (DLT)
    """
    assert len(src_pts) == len(dst_pts) and len(src_pts) >= 4
    
    # Build the matrix A for DLT
    A = []
    for (x, y), (x_prime, y_prime) in zip(src_pts, dst_pts):
        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
    
    A = np.array(A)
    
    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    
    # Reshape into 3x3 matrix
    H = h.reshape(3, 3)
    
    # Normalize
    H = H / H[2, 2]
    
    return H

def is_edge_point(point, img_shape, edge_threshold=10):
    """
    Check if a point is near the image edge
    """
    height, width = img_shape[:2]
    x, y = point
    
    return (x < edge_threshold or 
            x > width - edge_threshold or 
            y < edge_threshold or 
            y > height - edge_threshold)

def ransac_homography(matched_pairs, img1_corners, img2_corners, current_merge, current_image, 
                     match_dir_RANSAC, i, threshold=5.0, max_iterations=1000, edge_threshold=10):
    matches = matched_pairs[0]
    best_inliers = []
    max_inliers = 0
    best_H = None
    
    src_pts = np.float32(img1_corners)
    dst_pts = np.float32(img2_corners)
    
    # Filter out edge matches
    valid_matches = []
    for idx, match in enumerate(matches):
        src_point = src_pts[match[0]]
        dst_point = dst_pts[match[1]]
        
        # Skip if either point is near the edge
        if (is_edge_point(src_point, current_merge.shape, edge_threshold) or 
            is_edge_point(dst_point, current_image.shape, edge_threshold)):
            continue
            
        valid_matches.append(match)
    
    if len(valid_matches) < 4:
        print(f"Warning: Not enough valid matches after edge filtering for image {i}")
        return [], None
    
    for _ in range(max_iterations):
        # Randomly select 4 matches from valid matches
        random_indices = np.random.choice(len(valid_matches), 4, replace=False)
        sample_matches = [valid_matches[idx] for idx in random_indices]
        
        # Get corresponding points
        src_sample = np.float32([src_pts[match[0]] for match in sample_matches])
        dst_sample = np.float32([dst_pts[match[1]] for match in sample_matches])
        
        # Calculate homography
        H = compute_homography(src_sample, dst_sample)
        
        if H is None:
            continue
        
        # Count inliers from valid matches
        inliers = []
        for idx, match in enumerate(valid_matches):
            src = src_pts[match[0]]
            dst = dst_pts[match[1]]
            
            # Transform source point
            src_homogeneous = np.array([src[0], src[1], 1])
            dst_projected = H @ src_homogeneous
            dst_projected = dst_projected / dst_projected[2]
            
            # Calculate error
            error = np.sqrt(np.sum((dst - dst_projected[:2]) ** 2))
            
            if error < threshold:
                inliers.append(idx)
        
        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_inliers = inliers
            best_H = H
    
    # Visualization code
    if best_H is not None and best_inliers:
        filtered_matches = [valid_matches[idx] for idx in best_inliers]
        keypoints1 = [cv2.KeyPoint(float(x), float(y), 1) for (x, y) in img1_corners]
        keypoints2 = [cv2.KeyPoint(float(x), float(y), 1) for (x, y) in img2_corners]
        good_matches = [cv2.DMatch(_idx1, _idx2, 0) for _idx1, _idx2 in filtered_matches]
        
        match_img = cv2.drawMatches(
            current_merge, keypoints1,
            current_image, keypoints2,
            good_matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(100, 255, 0),
            singlePointColor=(255, 100, 0),
            matchesThickness=1
        )
        
        text = f"Inliers: {len(filtered_matches)}/{len(valid_matches)} ({len(filtered_matches)/len(valid_matches)*100:.1f}%)"
        cv2.putText(match_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        match_filename = f'ransac_match_{i}.jpg'
        match_save_path = os.path.join(match_dir_RANSAC, match_filename)
        cv2.imwrite(match_save_path, match_img)
    
    return best_inliers, best_H


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


def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--path", type=str, help="Path to the main directory containing image sets")
    Parser.add_argument("--Threshold", default=5.0, type=float, help="RANSAC Threshold")
    Parser.add_argument("--Nmax", default=1000, type=int, help="RANSAC iterations")
    Parser.add_argument("--EdgeThresh", default=10, type=int, help="Edge Threshold for avoiding features matching from image edges")
    

    Args = Parser.parse_args()

    threshold = Args.Threshold
    nmax = Args.Nmax
    edgethresh = Args.EdgeThresh

    # Get the directory path from the command line

    main_directory_1=os.path.abspath(Args.path)
    main_directory = os.path.abspath('Phase1/Code')

    set_name = os.path.basename(main_directory_1)
    print(f"Processing images from {main_directory}")
    # Define the output directory for corner-detected images
    corner_dir = os.path.join(os.path.dirname(os.path.dirname(main_directory)), "corner", set_name)
    print(f"Output directory for corner-detected images: {corner_dir}")


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


# Read images from the folder and store them in a list images
    print(f"Reading images from {main_directory_1}")
    if not os.path.exists(main_directory_1):
        print(f"Error: Directory '{main_directory_1}' not found.")
        exit(1)   
    for image_name in sorted(os.listdir(main_directory_1)):
        image_path = os.path.join(main_directory_1, image_name)
        # Read image using OpenCV
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)  # Append image to list
            print(f"Loaded image[{len(images) - 1}] from {image_path}")
        else:
            print(f"Could not read {image_path}")
        
   

    for i in range(len(images)-1):
        if i == 0:
            current_merge = images[i]

        current_image = images[i+1]
        current_image_p= cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        current_merge_p= cv2.cvtColor(current_merge, cv2.COLOR_BGR2GRAY)

        # Apply Shi-Tomasi Corner Detection
        image_corners = cv2.goodFeaturesToTrack(current_image_p, maxCorners=2000, qualityLevel=0.01, minDistance=8)
        merge_corners = cv2.goodFeaturesToTrack(current_merge_p, maxCorners=2000, qualityLevel=0.01, minDistance=8)
        print(f"Detected {len(image_corners)} corners in image {i + 1}")
        print(f"Detected {len(merge_corners)} corners in the current merge")
        if i == 0:
            corner_image = images[i].copy()
            if merge_corners is not None:
                merge_corners = np.intp(merge_corners)
                for corner in merge_corners:
                    x, y = corner.ravel()
                    cv2.circle(corner_image, (x, y), 2, (0, 0, 255), -1) 
            save_path = corner_dir + '/corners_' + str(i+1) + '.png'
            cv2.imwrite(save_path, corner_image)

        corner_image = images[i+1].copy()
        
        if image_corners is not None:
            image_corners = np.intp(image_corners)     
            for corner in image_corners:
                x, y = corner.ravel()
                cv2.circle(corner_image, (x, y), 2, (0, 0, 255), -1) 
        save_path = corner_dir + '/corners_' + str(i+2) + '.png'
        cv2.imwrite(save_path, corner_image)

        corner_image = current_merge.copy()
        if merge_corners is not None:
            merge_corners = np.intp(merge_corners)
            for corner in merge_corners:
                x, y = corner.ravel()
                cv2.circle(corner_image, (x, y), 2, (0, 0, 255), -1) 
        save_path = corner_dir + '/corners_merge_' + str(i+1) + '.png'
        cv2.imwrite(save_path, corner_image)

        num_best = 800
        marked_image_corners, selected_corners_image_corners = ANMS(image_corners, num_best, current_image)
        marked_merge_corners, selected_corners_merge_corners = ANMS(merge_corners, num_best, current_merge)
        print(f"Selected {len(selected_corners_image_corners)} corners in image {i + 1}")
        print(f"Selected {len(selected_corners_merge_corners)} corners in the current merge")
        if i == 0:
            save_path_merge_corner = os.path.join(corner_dir_ANMS, 'anms_' + str(i+1) + '.png')
            cv2.imwrite(save_path_merge_corner, marked_merge_corners)
            
        save_path_image_corner = os.path.join(corner_dir_ANMS, 'anms_' + str(i+2) + '.png')
        save_path_merge_corner = os.path.join(corner_dir_ANMS, 'anms_merge_' + str(i+1) + '.png')
        cv2.imwrite(save_path_image_corner, marked_image_corners)
        cv2.imwrite(save_path_merge_corner, marked_merge_corners)

        image_FD, FD_image_current=features(current_image, selected_corners_image_corners)
        merge_FD, FD_image_merge=features(current_merge, selected_corners_merge_corners)

        if i == 0:
            save_path = os.path.join(Feature, 'FD_' + str(i+1) + '.png')
            cv2.imwrite(save_path, FD_image_merge)

        save_path = os.path.join(Feature,'FD_' + str(i+2) + '.png')
        cv2.imwrite(save_path, FD_image_current)
        save_path = os.path.join(Feature, 'FD_merge' + str(i+1) + '.png')
        cv2.imwrite(save_path,FD_image_merge)
    
        print('loop :', i)
        matched_pairs = match_features(current_merge,current_image,image_FD,merge_FD,selected_corners_image_corners,selected_corners_merge_corners,match_dir,i)
        print('Feature matching is completed.')

        inliers, H = ransac_homography(matched_pairs, selected_corners_merge_corners, selected_corners_image_corners, current_merge,
                            current_image, match_dir_RANSAC, i, threshold=threshold, max_iterations=nmax, edge_threshold=edgethresh)
        if len(inliers)<11:
            print("Not enough inliers")
            continue
        
        else:
            print("Homography matrix: ", H)
            if len(images)>3:
                if i<len(images)//2:

                    result = stitch_images(current_image, current_merge, H)
                else:
                    result = stitch_images(current_merge, current_image, np.linalg.inv(H))
                result = cv2.resize(result, None, fx=0.9, fy=0.9, interpolation=cv2.INTER_AREA)
            else:
                result = stitch_images(current_image, current_merge, H)

            save_path = os.path.join(pano_dir, 'my_pano_' + str(i+1) + '.png')
            cv2.imwrite(save_path,result)

            current_merge = np.uint8(result)

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