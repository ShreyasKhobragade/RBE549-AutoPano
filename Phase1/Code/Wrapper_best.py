# #!/usr/bin/evn python

# """
# RBE/CS Fall 2022: Classical and Deep Learning Approaches for
# Geometric Computer Vision
# Project 1: MyAutoPano: Phase 2 Starter Code


# Author(s):
# Lening Li (lli4@wpi.edu)
# Teaching Assistant in Robotics Engineering,
# Worcester Polytechnic Institute
# """

# # Code starts here:
# # python load_images.py data/train/set1

# import numpy as np
# import cv2

# # Add any python libraries here
# import argparse
# import os

# # Path to the main directory containing image sets

# import numpy as np


# def ANMS(corners, num_best, original_image):
#     """
#     Performs Adaptive Non-Maximal Suppression to select well-distributed corner points
#     and visualizes them on the image.
    
#     Args:
#         corners: Numpy array of corner points from cv2.goodFeaturesToTrack
#         num_best: Number of best corners needed
#         original_image: Original image to draw corners on
        
#     Returns:
#         marked_image: Image with ANMS corners drawn
#         selected_corners: Numpy array of selected corner points
#     """
#     # Create a copy of the original image for drawing
#     marked_image = original_image.copy()
    
#     # Convert corners to a more manageable format
#     corners = corners.reshape(-1, 2)  # Convert to Nx2 array
#     N_strong = len(corners)
    
#     if N_strong == 0:
#         return marked_image, []
        
#     # Initialize r array
#     r = np.full(N_strong, np.inf)
    
#     # Calculate r for each point
#     for i in range(N_strong):
#         x_i, y_i = corners[i]
        
#         for j in range(N_strong):
#             if i == j:
#                 continue
                
#             x_j, y_j = corners[j]
            
#             # Calculate Euclidean distance
#             ED = (x_j - x_i)**2 + (y_j - y_i)**2
            
#             # Update r[i] if this distance is smaller
#             if ED < r[i]:
#                 r[i] = ED
    
#     # Create array of (r, index) pairs for sorting
#     corners_with_r = [(r[i], i) for i in range(N_strong)]
    
#     # Sort by r in descending order
#     corners_with_r.sort(reverse=True)
    
#     # Take top num_best points
#     selected_indices = [idx for _, idx in corners_with_r[:num_best]]
#     selected_corners = corners[selected_indices]
    
#     # Draw the selected corners on the image
#     # Draw all original corners in red
#     for corner in corners:
#         x, y = np.intp(corner)
#         cv2.circle(marked_image, (x, y), 4, (0, 0, 255), 2)  # Red circles
    
#     # Draw selected corners in green with larger radius
#     for corner in selected_corners:
#         x, y = np.intp(corner)
#         cv2.circle(marked_image, (x, y), 1, (0, 255, 0), -1)  # Green circles
        
#     # Add text showing number of corners
#     cv2.putText(marked_image, f'Total corners: {N_strong}', (10, 30), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#     cv2.putText(marked_image, f'Selected corners: {len(selected_corners)}', (10, 60), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
#     # Return corners in the same format as input for compatibility
#     selected_corners = selected_corners.reshape(-1, 2)
    
#     return marked_image, selected_corners

# def features(image, corners):
#     """
#     Generate feature descriptors for given corner points.
    
#     Args:
#         image: Input image (grayscale)
#         corners: Nx1x2 array of corner points from ANMS
        
#     Returns:
#         descriptors: Array of feature descriptors for each corner
#     """
#     # Convert to grayscale if needed
#     FD_image=image.copy()
#     if len(image.shape) == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#     # Reshape corners to Nx2
#     corners = corners.reshape(-1, 2)
    
#     # Parameters
#     patch_size = 41
#     output_size = 8
#     half_patch = patch_size // 2
    
#     # Initialize descriptor array
#     num_corners = len(corners)
#     descriptors = np.zeros((num_corners, output_size * output_size))
    
#     # For each corner point
#     for idx, corner in enumerate(corners):
#         x, y = np.intp(corner)
        
#         # Handle boundary cases
#         # Get patch bounds
#         y_start = max(y - half_patch, 0)
#         y_end = min(y + half_patch + 1, image.shape[0])
#         x_start = max(x - half_patch, 0)
#         x_end = min(x + half_patch + 1, image.shape[1])
        
#         # Extract patch
#         patch = np.zeros((patch_size, patch_size))
#         actual_patch = image[y_start:y_end, x_start:x_end]
        
#         # Calculate where to place the actual patch in the zero-padded patch
#         patch_y_start = half_patch - (y - y_start)
#         patch_x_start = half_patch - (x - x_start)
#         patch[patch_y_start:patch_y_start + actual_patch.shape[0],
#               patch_x_start:patch_x_start + actual_patch.shape[1]] = actual_patch
        
#         # Apply Gaussian blur
#         blurred_patch = cv2.GaussianBlur(patch, (3, 3), 0)
        
#         # Subsample to 8x8
#         # Calculate indices for evenly spaced samples
#         row_indices = np.linspace(0, patch_size-1, output_size, dtype=int)
#         col_indices = np.linspace(0, patch_size-1, output_size, dtype=int)
#         subsampled = blurred_patch[row_indices][:, col_indices]
        
        
#         cv2.circle(FD_image, (x, y), 2, (255, 0, 0), -1)  # Red circles
#         cv2.rectangle(FD_image, (x-half_patch,y+half_patch), (x+half_patch,y-half_patch), (255, 255, 0), 1)
#         # Reshape to 64x1
#         feature_vector = subsampled.reshape(-1)
        
#         # Standardize to zero mean and unit variance
#         if np.std(feature_vector) != 0:  # Avoid division by zero
#             feature_vector = (feature_vector - np.mean(feature_vector)) / np.std(feature_vector)
        
#         # Store the feature vector
#         descriptors[idx] = feature_vector
        
		
    
#     return descriptors, FD_image

# def match_features(image_data, match_dir):

    
#     # Get sorted image names to maintain order
# 	image_names = sorted(list(image_data.keys()))
# 	matched_pairs = []
    
#     # Process consecutive pairs
# 	for i in range(len(image_names)):
# 		for j in range(i+1,len(image_names)):
# 			img1_name = image_names[i]
# 			img2_name = image_names[j]
			
# 			# Get feature data from dictionary
# 			img1_features = image_data[img1_name]['features']
# 			img2_features = image_data[img2_name]['features']
# 			img1_corners = image_data[img1_name]['anms_corners']
# 			img2_corners = image_data[img2_name]['anms_corners']
			
# 			keypoints1 = [cv2.KeyPoint(float(x), float(y), 1) for (x, y) in img1_corners]
# 			keypoints2 = [cv2.KeyPoint(float(x), float(y), 1) for (x,y) in img2_corners]
			
# 			matches = []
# 			# For each feature in first image
# 			for idx1, feat1 in enumerate(img1_features):
# 				distances = []
# 				# Compare with all features in second image
# 				for idx2, feat2 in enumerate(img2_features):
# 					dist = np.sum((feat1 - feat2) ** 2)  # SSD
# 					distances.append((dist, idx2))
				
# 				# Sort distances
# 				distances.sort(key=lambda x: x[0])
				
# 				# Apply ratio test
# 				if len(distances) >= 2:
# 					ratio = distances[0][0] / distances[1][0]
# 					if ratio < 0.8:  # Lowe's ratio test threshold
# 						matches.append((idx1, distances[0][1]))
			
# 			# Visualize matches
# 			img1 = image_data[img1_name]['image']
# 			img2 = image_data[img2_name]['image']

# 			# Convert matches to cv2.DMatch format
# 			good_matches = [cv2.DMatch(_idx1, _idx2, 0) for _idx1, _idx2 in matches]

# 			# Draw matches
# 			match_img = cv2.drawMatches(img1, keypoints1, 
# 										img2, keypoints2,
# 										good_matches, None,
# 										flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
# 										matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),matchesThickness=2)

# 			match_filename =f'match_{os.path.splitext(img1_name)[0]}_{os.path.splitext(img2_name)[0]}.jpg'
# 			match_save_path = os.path.join(match_dir, match_filename)
# 			# Save match visualization
# 			cv2.imwrite(match_save_path, match_img)
# 			matched_pairs.append((img1_name, img2_name, matches))
    
# 	return matched_pairs
# def compute_homography(points1, points2):
#     """
#     Compute homography matrix from corresponding points
#     """
#     if len(points1) < 4:
#         return None
        
#     A = []
#     for i in range(len(points1)):
#         x, y = points1[i]
#         u, v = points2[i]
#         A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
#         A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    
#     A = np.array(A)
    
#     # Solve using SVD
#     _, _, Vt = np.linalg.svd(A)
    
#     # Get the last row of Vt and reshape it to 3x3 matrix
#     H = Vt[-1].reshape(3, 3)
    
#     # Normalize H
#     H = H / H[2, 2]
    
#     return H

# def apply_homography(H, points):
#     """
#     Apply homography to points
#     """
#     # Convert to homogeneous coordinates
#     points_homog = np.hstack((points, np.ones((len(points), 1))))
    
#     # Apply homography
#     transformed_points = np.dot(H, points_homog.T).T
    
#     # Convert back to inhomogeneous coordinates
#     transformed_points = transformed_points[:, :2] / transformed_points[:, 2:]
#     return transformed_points

# def match_features_with_ransac(matched_pairs, image_data, match_dir, ransac_thresh, max_iters):
#     """
#     Apply RANSAC to filter matched features with improved homography estimation
    
#     Args:
#         matched_pairs: List of tuples (img1_name, img2_name, matches)
#         image_data: Dictionary containing image data
#         match_dir: Directory to save visualizations
#         ransac_thresh: Distance threshold for inlier classification
#         max_iters: Maximum RANSAC iterations
#     Returns:
#         List of tuples (img1_name, img2_name, filtered_matches, homography)
#     """
#     def normalize_points(points):
#         """Normalize points for better numerical stability"""
#         centroid = np.mean(points, axis=0)
#         dist = np.sqrt(np.sum((points - centroid) ** 2, axis=1))
#         scale = np.sqrt(2) / np.mean(dist)
#         T = np.array([[scale, 0, -scale * centroid[0]],
#                      [0, scale, -scale * centroid[1]],
#                      [0, 0, 1]])
#         points_homog = np.column_stack([points, np.ones(len(points))])
#         normalized_points = (T @ points_homog.T).T[:, :2]
#         return normalized_points, T

#     def compute_homography_normalized(points1, points2):
#         """Compute homography with normalized coordinates"""
#         # Normalize points
#         norm_points1, T1 = normalize_points(points1)
#         norm_points2, T2 = normalize_points(points2)
        
#         # Compute homography with normalized points
#         A = []
#         for i in range(len(norm_points1)):
#             x, y = norm_points1[i]
#             u, v = norm_points2[i]
#             A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
#             A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
        
#         A = np.array(A)
#         _, _, Vt = np.linalg.svd(A)
#         H_norm = Vt[-1].reshape(3, 3)
        
#         # Denormalize
#         H = np.linalg.inv(T2) @ H_norm @ T1
#         return H / H[2, 2]

#     ransac_results = []
    
#     for img1_name, img2_name, matches in matched_pairs:
#         if len(matches) < 4:
#             print(f"Skipping {img1_name}-{img2_name}: Not enough matches ({len(matches)})")
#             continue
            
#         # Get corner points
#         img1_corners = image_data[img1_name]['anms_corners']
#         img2_corners = image_data[img2_name]['anms_corners']
        
#         # Convert matches to point arrays
#         points1 = np.float32([img1_corners[m[0]] for m in matches])
#         points2 = np.float32([img2_corners[m[1]] for m in matches])
        
#         best_inliers = []
#         best_H = None
#         best_score = 0
        
#         # RANSAC loop
#         for _ in range(max_iters):
#             # Random sample 4 points
#             if len(matches) < 4:
#                 continue
#             sample_idx = np.random.choice(len(matches), 4, replace=False)
#             sample_points1 = points1[sample_idx]
#             sample_points2 = points2[sample_idx]
            
#             # Compute homography with normalized coordinates
#             try:
#                 H = compute_homography_normalized(sample_points1, sample_points2)
                
#                 # Transform all points
#                 points1_homog = np.column_stack([points1, np.ones(len(points1))])
#                 transformed_points = (H @ points1_homog.T).T
#                 transformed_points = transformed_points[:, :2] / transformed_points[:, 2:]
                
#                 # Compute symmetric transfer error
#                 errors = np.sqrt(np.sum((transformed_points - points2)**2, axis=1))
#                 inliers = errors < ransac_thresh
                
#                 # Compute score (considering both number of inliers and their error)
#                 score = np.sum(inliers) - np.sum(errors[inliers]) / ransac_thresh
                
#                 if score > best_score:
#                     best_score = score
#                     best_inliers = inliers
#                     best_H = H
                    
#                     # Early termination if we found a very good model
#                     inlier_ratio = np.sum(inliers) / len(matches)
#                     if inlier_ratio > 0.8:  # 80% inliers
#                         break
                        
#             except np.linalg.LinAlgError:
#                 continue
        
#         if best_H is not None and len(best_inliers) >= 4:
#             # Refine homography using all inliers
#             inlier_points1 = points1[best_inliers]
#             inlier_points2 = points2[best_inliers]
#             refined_H = compute_homography_normalized(inlier_points1, inlier_points2)
            
#             # Filter matches using refined model
#             filtered_matches = [m for m, is_inlier in zip(matches, best_inliers) if is_inlier]
            
#             # Visualize RANSAC matches
#             keypoints1 = [cv2.KeyPoint(float(x), float(y), 1) for (x, y) in img1_corners]
#             keypoints2 = [cv2.KeyPoint(float(x), float(y), 1) for (x, y) in img2_corners]
#             good_matches = [cv2.DMatch(_idx1, _idx2, 0) for _idx1, _idx2 in filtered_matches]
            
#             match_img = cv2.drawMatches(
#                 image_data[img1_name]['image'], keypoints1,
#                 image_data[img2_name]['image'], keypoints2,
#                 good_matches, None,
#                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
#                 matchColor=(0, 255, 0),
#                 singlePointColor=(255, 0, 0),
#                 matchesThickness=2
#             )
            
#             # Add text with match statistics
#             text = f"Inliers: {len(filtered_matches)}/{len(matches)} ({len(filtered_matches)/len(matches)*100:.1f}%)"
#             cv2.putText(match_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
#                        1, (0, 255, 0), 2)
            
#             match_filename = f'ransac_match_{os.path.splitext(img1_name)[0]}_{os.path.splitext(img2_name)[0]}.jpg'
#             match_save_path = os.path.join(match_dir, match_filename)
#             cv2.imwrite(match_save_path, match_img)
            
#             ransac_results.append((img1_name, img2_name, filtered_matches, refined_H))
#             print(f"Matched {img1_name}-{img2_name}: {len(filtered_matches)} inliers from {len(matches)} matches")
    
#     return ransac_results

# # def match_features_with_ransac(image_data, match_dir, ransac_thresh, max_iters=10):
# #     """
# #     Modified match_features function that includes RANSAC for homography estimation
# #     """
# #     image_names = sorted(list(image_data.keys()))
# #     matched_pairs = []
# #     max_iters = 10

# #     ransac_thresh = 5.0
    
# #     for i in range(len(image_names)):
# #         for j in range(i+1, len(image_names)):
# #             img1_name = image_names[i]
# #             img2_name = image_names[j]
            
# #             # Get feature data
# #             img1_features = image_data[img1_name]['features']
# #             img2_features = image_data[img2_name]['features']
# #             img1_corners = image_data[img1_name]['anms_corners']
# #             img2_corners = image_data[img2_name]['anms_corners']
            
# #             # Initial feature matching
# #             matches = []
# #             for idx1, feat1 in enumerate(img1_features):
# #                 distances = []
# #                 for idx2, feat2 in enumerate(img2_features):
# #                     dist = np.sum((feat1 - feat2) ** 2)
# #                     distances.append((dist, idx2))
                
# #                 distances.sort(key=lambda x: x[0])
# #                 if len(distances) >= 2:
# #                     ratio = distances[0][0] / distances[1][0]
# #                     if ratio < 0.8:
# #                         matches.append((idx1, distances[0][1]))
            
# #             if len(matches) < 4:
# #                 continue
                
# #             # Convert matches to point arrays for RANSAC
# #             points1 = np.float32([img1_corners[m[0]] for m in matches])
# #             points2 = np.float32([img2_corners[m[1]] for m in matches])
            
# #             # RANSAC implementation
# #             best_inliers = []
# #             best_H = None
            
# #             for a in range(max_iters):
# #                 # 1. Randomly select 4 points
# #                 rand_indices = np.random.choice(len(matches), 4, replace=False)
# #                 sample_points1 = points1[rand_indices]
# #                 sample_points2 = points2[rand_indices]
                
# #                 # 2. Compute homography
# #                 H = compute_homography(sample_points1, sample_points2)
# #                 if H is None:
# #                     continue
                
# #                 # 3. Transform all points and compute SSD
# #                 transformed_points = apply_homography(H, points1)
                
# #                 # 4. Count inliers
# #                 distances = np.sqrt(np.sum((transformed_points - points2)**2, axis=1))
# #                 inliers = distances < ransac_thresh
# #                 inlier_count = np.sum(inliers)
                
# #                 # Update best model if we found more inliers
# #                 if inlier_count > len(best_inliers):
# #                     best_inliers = inliers
# #                     best_H = H
                
# #                 # Optional early termination
# #                 if inlier_count > len(matches) * 0.9:  # 90% inliers
# #                     break
            
# #             # Filter matches using best inliers
# #             filtered_matches = [m for m, is_inlier in zip(matches, best_inliers) if is_inlier]
            
# #             # Visualize matches
# #             keypoints1 = [cv2.KeyPoint(float(x), float(y), 1) for (x, y) in img1_corners]
# #             keypoints2 = [cv2.KeyPoint(float(x), float(y), 1) for (x, y) in img2_corners]
            
# #             good_matches = [cv2.DMatch(_idx1, _idx2, 0) for _idx1, _idx2 in filtered_matches]
            
# #             img1 = image_data[img1_name]['image']
# #             img2 = image_data[img2_name]['image']
            
# #             match_img = cv2.drawMatches(img1, keypoints1, 
# #                                       img2, keypoints2,
# #                                       good_matches, None,
# #                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
# #                                       matchColor=(0, 255, 0),
# #                                       singlePointColor=(255, 0, 0),
# #                                       matchesThickness=2)
            
# #             match_filename = f'ransac_match_{os.path.splitext(img1_name)[0]}_{os.path.splitext(img2_name)[0]}.jpg'
# #             match_save_path = os.path.join(match_dir, match_filename)
# #             cv2.imwrite(match_save_path, match_img)
            
# #             matched_pairs.append((img1_name, img2_name, filtered_matches, best_H))
    
# #     return matched_pairs
# import numpy as np
# import cv2
# def compute_output_size(images, homographies):
#     """
#     Compute the size of the output panorama given the input images and homographies
#     """
#     def warp_corners(H, corners):
#         # Convert to homogeneous coordinates
#         corners_homog = np.column_stack([corners, np.ones(len(corners))])
#         warped_corners = (H @ corners_homog.T).T
#         warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
#         return warped_corners

#     h, w = images[0].shape[:2]
#     corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    
#     # Initialize with corners of the first image
#     all_corners = [corners]
    
#     # Warp corners of all other images
#     for H in homographies[1:]:
#         warped_corners = warp_corners(H, corners)
#         all_corners.append(warped_corners)
    
#     # Find min and max coordinates
#     all_corners = np.vstack(all_corners)
#     min_x, min_y = np.floor(np.min(all_corners, axis=0)).astype(int)
#     max_x, max_y = np.ceil(np.max(all_corners, axis=0)).astype(int)
    
#     return (max_y - min_y, max_x - min_x), (min_x, min_y)

# def create_mask(img):
#     """
#     Create a gradient mask for blending
#     """
#     mask = np.zeros(img.shape[:2], dtype=np.float32)
#     h, w = mask.shape
    
#     # Create horizontal gradient
#     for i in range(w):
#         mask[:, i] = i / w
        
#     return mask
# def create_panorama(image_data, ransac_results):
#     """
#     Create panorama from image data and RANSAC results
    
#     Args:
#         image_data: Dictionary containing image information
#         ransac_results: List of tuples (img1_name, img2_name, matches, homography)
#     Returns:
#         Panorama image
#     """
#     # Get sorted list of image names
#     image_names = sorted(list(image_data.keys()))
    
#     # Get list of images in order
#     images = [image_data[name]['image'] for name in image_names]
    
#     # Stitch images
#     panorama = stitch_images(images, ransac_results)
    
#     # Crop black borders
#     if panorama is not None:
#         # Convert to grayscale for finding non-black regions
#         gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        
#         # Convert to 8-bit unsigned integer
#         gray = (gray * 255).astype(np.uint8)
        
#         # Create binary threshold
#         _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
#         # Ensure threshold is 8-bit unsigned integer
#         thresh = thresh.astype(np.uint8)
        
#         # Find contours
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         if contours:
#             # Find bounding rectangle of all non-black pixels
#             x, y, w, h = cv2.boundingRect(np.vstack(contours))
#             panorama = panorama[y:y+h, x:x+w]
    
#     return panorama

# def create_mask(img):
#     """
#     Create a gradient mask for blending
#     """
#     mask = np.zeros(img.shape[:2], dtype=np.float32)
#     h, w = mask.shape
    
#     # Create horizontal gradient
#     for i in range(w):
#         mask[:, i] = i / w
        
#     return mask

# def blend_images(base_img, overlay_img, mask):
#     """
#     Blend two images using a mask
#     """
#     if base_img is None:
#         return overlay_img
#     if overlay_img is None:
#         return base_img
        
#     mask = np.expand_dims(mask, axis=2) if len(mask.shape) == 2 else mask
#     blended = base_img * (1 - mask) + overlay_img * mask
#     return blended.astype(np.uint8)  # Ensure output is uint8

# def stitch_images(images, ransac_results):
#     """
#     Stitch multiple images together using the RANSAC results
#     """
#     # Convert ransac_results to a more usable format
#     homographies = {}
#     for img1_name, img2_name, _, H in ransac_results:
#         homographies[(img1_name, img2_name)] = H
    
#     # Get unique image names in order
#     image_names = []
#     for img1_name, img2_name, _, _ in ransac_results:
#         if img1_name not in image_names:
#             image_names.append(img1_name)
#         if img2_name not in image_names:
#             image_names.append(img2_name)
    
#     n_images = len(images)
    
#     # Use the middle image as reference
#     mid_idx = n_images // 2
#     cumulative_H = [np.eye(3)]  # Identity for middle image
    
#     # Forward direction homographies
#     curr_H = np.eye(3)
#     for i in range(mid_idx + 1, n_images):
#         prev_name = image_names[i-1]
#         curr_name = image_names[i]
#         if (prev_name, curr_name) in homographies:
#             H = homographies[(prev_name, curr_name)]
#         else:
#             H = np.linalg.inv(homographies[(curr_name, prev_name)])
#         curr_H = H @ curr_H
#         cumulative_H.append(curr_H.copy())
    
#     # Backward direction homographies
#     curr_H = np.eye(3)
#     backward_H = [np.eye(3)]
#     for i in range(mid_idx - 1, -1, -1):
#         next_name = image_names[i+1]
#         curr_name = image_names[i]
#         if (curr_name, next_name) in homographies:
#             H = homographies[(curr_name, next_name)]
#         else:
#             H = np.linalg.inv(homographies[(next_name, curr_name)])
#         curr_H = H @ curr_H
#         backward_H.append(curr_H.copy())
    
#     # Combine homographies
#     cumulative_H = list(reversed(backward_H[1:])) + cumulative_H
    
#     # Compute output size
#     output_size, offset = compute_output_size(images, cumulative_H)
    
#     # Create translation matrix
#     T = np.array([[1, 0, -offset[0]], 
#                   [0, 1, -offset[1]], 
#                   [0, 0, 1]])
    
#     # Initialize output image
#     panorama = None
#     accumulated_mask = None
    
#     # Warp and blend each image
#     for idx, (image, H) in enumerate(zip(images, cumulative_H)):
#         # Apply translation to homography
#         H = T @ H
        
#         # Warp image
#         warped = cv2.warpPerspective(image, H, (output_size[1], output_size[0]))
        
#         # Create and warp mask
#         mask = create_mask(image)
#         warped_mask = cv2.warpPerspective(mask, H, (output_size[1], output_size[0]))
        
#         # Update accumulated mask
#         if accumulated_mask is None:
#             accumulated_mask = warped_mask
#         else:
#             overlap = (accumulated_mask > 0) & (warped_mask > 0)
#             if np.any(overlap):
#                 warped_mask[overlap] = np.maximum(warped_mask[overlap], accumulated_mask[overlap])
#             accumulated_mask = np.maximum(accumulated_mask, warped_mask)
        
#         # Blend images
#         panorama = blend_images(panorama, warped, warped_mask)
    
#     return panorama
# def main():
#     # Add any Command Line arguments here
#     Parser = argparse.ArgumentParser()
#     # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

#     # Args = Parser.parse_args()
#     # NumFeatures = Args.NumFeatures
#     Parser = argparse.ArgumentParser()
#     Parser.add_argument("--path", type=str, help="Path to the main directory containing image sets")
#     Parser.add_argument("--Threshold", type=float, help="RANSAC Threshold")
#     Parser.add_argument("--Nmax", type=int, help="RANSAC iterations")

#     Args = Parser.parse_args()

#     # Get the directory path from the command line
#     main_directory = os.path.abspath(Args.path)
#     set_name = os.path.basename(main_directory)

# # Define the output directory for corner-detected images
#     corner_dir = os.path.join(os.path.dirname(os.path.dirname(main_directory)), "corner", set_name)

#     # Create the output folder if it doesn't exist
#     os.makedirs(corner_dir, exist_ok=True)
#     corner_dir_ANMS = os.path.join(os.path.dirname(os.path.dirname(main_directory)), "corner_ANMS", set_name)

#     match_dir = os.path.join(os.path.dirname(os.path.dirname(main_directory)), "match", set_name)
#     os.makedirs(match_dir, exist_ok=True)
#     match_dir_RANSAC = os.path.join(os.path.dirname(os.path.dirname(main_directory)), "ransac_match", set_name)
#     os.makedirs(match_dir_RANSAC, exist_ok=True)
#     pano_dir = os.path.join(os.path.dirname(os.path.dirname(main_directory)), "Pano", set_name)
#     os.makedirs(pano_dir, exist_ok=True)

#     # Create the output folder if it doesn't exist
#     os.makedirs(corner_dir_ANMS, exist_ok=True)

#     Feature = os.path.join(os.path.dirname(os.path.dirname(main_directory)), "Feature", set_name)

#     # Create the output folder if it doesn't exist
#     os.makedirs(Feature, exist_ok=True)

#     # List to store images
#     images = []
#     image_corners=[]
#     ANMS_corners = []
#     ANMS_Images = []
#     Features_D=[]
#     image_data = {}

# # Iterate over each folder (image set)
#  # Ensure it's a directory
#     print(f"Reading images from {main_directory}")

#     if not os.path.exists(main_directory):
#         print(f"Error: Directory '{main_directory}' not found.")
#         exit(1)

#             # Iterate over each image in the folder
#     for image_name in os.listdir(main_directory):
#         image_path = os.path.join(main_directory, image_name)

#         # Read image using OpenCV
#         image = cv2.imread(image_path)

#         if image is not None:
#             images.append(image)  # Append image to list
#             print(f"Loaded image[{len(images) - 1}] from {image_path}")
#         else:
#             print(f"Could not read {image_path}")
#         image_data[image_name] = {
            
#             'image': image,
#             'anms_corners': None,
#             'features': None
#         }

#         image_corner_preprocessing= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply Shi-Tomasi Corner Detection
#         corners = cv2.goodFeaturesToTrack(image_corner_preprocessing, maxCorners=800, qualityLevel=0.001, minDistance=15)
#         corner_image = image.copy()
#         if corners is not None:
#             corners = np.intp(corners)  # Convert to integer values
        
#         # Draw detected corners ON the original RGB image
#             for corner in corners:
#                 x, y = corner.ravel()
#                 cv2.circle(corner_image, (x, y), 5, (0, 255, 0), -1)  # Mark in Green (BGR: (0, 255, 0))
#             image_corners.append(corner_image)

#         save_path = os.path.join(corner_dir, image_name)
#         cv2.imwrite(save_path, corner_image)
#         # Show the result
#         # cv2.imshow(f"Shi-Tomasi Corners - {image_name}", image_corners[-1])
#         cv2.waitKey(500)  # Display each image for 500ms
            

#         num_best = 250
#         marked_image, selected_corners = ANMS(corners, num_best, image)
#         ANMS_Images.append(marked_image)
#         ANMS_corners.append(selected_corners)
#         image_data[image_name]['anms_corners'] = selected_corners
#         save_path = os.path.join(corner_dir_ANMS, image_name)
#         cv2.imwrite(save_path, marked_image)
            
#         Fd, FD_image=features(image, selected_corners)
#         Features_D.append(Fd)
#         image_data[image_name]['features'] = Fd
#         save_path = os.path.join(Feature, image_name)
#         cv2.imwrite(save_path, FD_image)
    
#     match_pairs = match_features(image_data,match_dir)
#     match_pairs_ransac = match_features_with_ransac(match_pairs,image_data, match_dir_RANSAC, ransac_thresh=120.0, max_iters=1000)
#     panorama = create_panorama(image_data, match_pairs_ransac)


# # Save the panorama
#     if panorama is not None:
#         cv2.imwrite('mypano.png', panorama)
#     else:
#         print("Failed to create panorama")

# # Save the panorama
#     pano_loc=os.path.join(pano_dir, f"{set_name}_pano.jpg")
#     cv2.imwrite(pano_loc, panorama)
        
        
        

        
        
    

#     # Display images (optional)
#     # for i, img in enumerate(images):
#     #     cv2.imshow(f"Image {i}", img)
#     #     cv2.waitKey(500)  # Display each image for 500ms

#     # cv2.destroyAllWindows()




# if __name__ == "__main__":
#     main()
	
# """
# Read a set of images for Panorama stitching
# """

# """
# Corner Detection
# Save Corner detection output as corners.png
# """

# """
# Perform ANMS: Adaptive Non-Maximal Suppression
# Save ANMS output as anms.png
# """

# """
# Feature Descriptors
# Save Feature Descriptor output as FD.png
# """

# """
# Feature Matching
# Save Feature Matching output as matching.png
# """

# """
# Refine: RANSAC, Estimate Homography
# """

# """
# Image Warping + Blending
# Save Panorama output as mypano.png
# """



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
    """
    Performs Adaptive Non-Maximal Suppression to select well-distributed corner points
    and visualizes them on the image.
    
    Args:
        corners: Numpy array of corner points from cv2.goodFeaturesToTrack
        num_best: Number of best corners needed
        original_image: Original image to draw corners on
        
    Returns:
        marked_image: Image with ANMS corners drawn
        selected_corners: Numpy array of selected corner points
    """
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
    cv2.putText(marked_image, f'Selected corners: {len(selected_corners)}', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Return corners in the same format as input for compatibility
    selected_corners = selected_corners.reshape(-1, 2)
    
    return marked_image, selected_corners

def features(image, corners):
    """
    Generate feature descriptors for given corner points.
    
    Args:
        image: Input image (grayscale)
        corners: Nx1x2 array of corner points from ANMS
        
    Returns:
        descriptors: Array of feature descriptors for each corner
    """
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

    
    # Get sorted image names to maintain order
	#mage_names = sorted(list(image_data.keys()))
    matched_pairs = []
    
    # Process consecutive pairs

    # img1_name = image
    # img2_name = image_names[j]
    
    # Get feature data from dictionary
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
    """
    Compute homography matrix from corresponding points
    """
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
    """
    Apply RANSAC to filter matched features with improved homography estimation
    
    Args:
        matched_pairs: List of tuples (img1_name, img2_name, matches)
        image_data: Dictionary containing image data
        match_dir: Directory to save visualizations
        ransac_thresh: Distance threshold for inlier classification
        max_iters: Maximum RANSAC iterations
    Returns:
        List of tuples (img1_name, img2_name, filtered_matches, homography)
    """
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

# def homogeneous_coordinate(coordinate):
#     x = coordinate[0]/coordinate[2]
#     y = coordinate[1]/coordinate[2]
#     return x, y

# def warp(image, homography):
#     print("Warping is started.")

#     image_array = np.array(image)
#     row_number, column_number = int(image_array.shape[0]), int(image_array.shape[1])

#     up_left_cor_x, up_left_cor_y = homogeneous_coordinate(np.dot(homography, [[0],[0],[1]]))
#     up_right_cor_x, up_right_cor_y = homogeneous_coordinate(np.dot(homography, [[column_number-1],[0],[1]]))
#     low_left_cor_x, low_left_cor_y = homogeneous_coordinate(np.dot(homography, [[0],[row_number-1],[1]]))
#     low_right_cor_x, low_right_cor_y = homogeneous_coordinate(np.dot(homography, [[column_number-1],[row_number-1],[1]]))

#     x_values = [up_left_cor_x, up_right_cor_x, low_right_cor_x, low_left_cor_x]
#     y_values = [up_left_cor_y, up_right_cor_y, low_left_cor_y,  low_right_cor_y]
#     print("x_values: ", x_values, "\n y_values: ", y_values)

#     offset_x = math.floor(np.min(x_values))
#     offset_y = math.floor(np.min(y_values))
#     print("offset_x: ", offset_x, "\t size_y: ", offset_x)

#     max_x = math.ceil(np.max(x_values))
#     max_y = math.ceil(np.max(y_values))

#     size_x = max_x - offset_x
#     size_y = max_y - offset_y
#     print("size_x: ", size_x, "\t size_y: ", size_y)

#     homography_inverse = np.linalg.inv(homography)
#     print("Homography inverse: ", "\n", homography_inverse)

#     result = 255*np.ones((size_y, size_x, 3))

#     for x in range(size_x):
#         for y in range(size_y):
#             point_xy = homogeneous_coordinate(np.dot(homography_inverse, [[x+offset_x], [y+offset_y], [1]]))
#             point_x = int(point_xy[0].item())
#             point_y = int(point_xy[1].item())

#             if (point_x >= 0 and point_x < column_number and point_y >= 0 and point_y < row_number):
#                 result[y, x, :] = image_array[point_y, point_x, :]

#     print("Warping is completed.")
#     return result, offset_x, offset_y




import numpy as np
import math

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
            current_merge = images[i]


        current_image = images[i+1]

        current_image_p= cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        current_merge_p= cv2.cvtColor(current_merge, cv2.COLOR_BGR2GRAY)

        # Apply Shi-Tomasi Corner Detection
        image_corners = cv2.goodFeaturesToTrack(current_image_p, maxCorners=800, qualityLevel=0.01, minDistance=8)
        merge_corners = cv2.goodFeaturesToTrack(current_merge_p, maxCorners=800, qualityLevel=0.01, minDistance=8)

        if i == 0:
            corner_image = images[i].copy()
            if merge_corners is not None:
                merge_corners = np.intp(merge_corners)
                for corner in merge_corners:
                    x, y = corner.ravel()
                    cv2.circle(corner_image, (x, y), 2, (0, 0, 255), -1)  # Mark in Green (BGR: (0, 255, 0))
            save_path = corner_dir + '/corner_' + str(i+1) + '.png'
            cv2.imwrite(save_path, corner_image)
        # else:
        #     corner_image = images[i].copy()
        #     if merge_corners is not None:
        #         merge_corners = np.intp(merge_corners)
        #         for corner in merge_corners:
        #             x, y = corner.ravel()
        #             cv2.circle(corner_image, (x, y), 2, (0, 0, 255), -1)  # Mark in Green (BGR: (0, 255, 0))
        #     save_path = corner_dir + '/corner_merge' + str(i+1) + '.png'
        #     cv2.imwrite(save_path, corner_image)
            

        corner_image = images[i+1].copy()
        
        if image_corners is not None:
            image_corners = np.intp(image_corners)  # Convert to integer values        
            for corner in image_corners:
                x, y = corner.ravel()
                cv2.circle(corner_image, (x, y), 2, (0, 0, 255), -1)  # Mark in Green (BGR: (0, 255, 0))
        save_path = corner_dir + '/corner_' + str(i+2) + '.png'
        cv2.imwrite(save_path, corner_image)

        # Perform ANMS


        num_best = 400
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
        match_pairs_ransac, H = match_features_with_ransac(matched_pairs,current_merge,current_image,image_FD,merge_FD,selected_corners_image_corners,selected_corners_merge_corners,match_dir_RANSAC, i,ransac_thresh=3.0, max_iters=2000)
        # match_pairs_ransac, H = match_features_with_ransac(matched_pairs,current_merge,current_image,image_FD,merge_FD,selected_corners_image_corners,selected_corners_merge_corners,match_dir_RANSAC, i,ransac_thresh=15.0, max_iters=1000)
# 

        warped_image_source1, source1_offset_x, source1_offset_y = warp(current_merge, H[0])
        #warped_image_source1, source1_offset_x, source1_offset_y = warp(current_image, np.linalg.inv(H[0]))
        image1 = Image.fromarray(warped_image_source1.astype('uint8'), 'RGB')
        image1.save("warped_image.jpg")
        #result = cv2.seamlessClone(current_image, current_merge, mask, center, cv2.MIXED_CLONE)
        result_blended = blending2images(current_image, warped_image_source1, source1_offset_x, source1_offset_y)
        # cv2.imshow("Blended Image", result_blended)
        image_final = Image.fromarray(result_blended.astype('uint8'), 'RGB')
        image_final.save("blended_image.jpg")
        print("Blended image is generated.")
        print(result_blended.shape)
        
        current_merge = np.uint8(result_blended)
        # current_merge = cv2.resize(current_merge, None, fx=0.9, fy=0.9, interpolation=cv2.INTER_LINEAR )

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