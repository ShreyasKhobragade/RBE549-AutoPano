import os
import cv2
import numpy as np
from PIL import Image
import csv
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
            perturb += np.random.randint(0,8)
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


if __name__ == "__main__":
    input_folder = "../Data/Train"
    output_folder = "../Data/TrainSet"
    
    params = {
        'num_patches': 10,
        'patch_size': (128, 128),
        'rho': 32,
        'edge_margin': 40,
        'resize': (320, 240)

    }
    
    process_dataset(input_folder, output_folder, **params)

    input_folder = "../Data/Val"
    output_folder = "../Data/ValSet"
    
    process_dataset(input_folder, output_folder, **params)
    
