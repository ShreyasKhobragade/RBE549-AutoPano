# RBE549-AutoPano Phase 2
# Deep Learning-Based Panorama Stitching

## Project Overview
This project implements advanced panorama stitching techniques using deep learning approaches, moving beyond traditional methods that rely on sequential pipelines of corner detection, feature extraction, matching, and RANSAC-based homography computation. By leveraging neural networks, we provide an end-to-end solution that unifies these multiple stages into a single robust framework, potentially reducing computational overhead while improving accuracy through learned features and transformations.

## Key Components

### To run the code navigate to "Group12_p1/Phase2/Code" location first
### To use pretrained checkpoints download the checkpoint folder from the link given at the bottom and add that folder to "Gropu12_p1/Phase2"
### 1. Data Generation
A comprehensive data generation pipeline that creates training datasets for both supervised and unsupervised learning approaches. This includes:
- Generation of image pairs with known homography transformations
- Creation of patches and corresponding ground truth data
- Automated labeling for supervised learning

To run data generation for this project, you need to execute:



```bash
python DataGen.py
```

This will run the data generation script that creates:
- Training and validation datasets
- Original and warped image patches
- Labels containing corner coordinates and image IDs
- Data stored in appropriate directories 

Here's how to set up the data structure for Phase 2:

1. Create Data folder in Phase2:
```
Phase2/
└── Data/
```

2. Add training and validation images:
```
Phase2/
└── Data/
    ├── Train/
    │   └── [training images]
    └── Val/
        └── [validation images]
```

3. Run the data generation script as mentioned above.

4. After running Datagen.py, the following structure will be created:
```
Phase2/
└── Data/
    ├── Train/
    ├── Val/
    ├── TrainSet/
    │   ├── original_patches/
    │   ├── warped_patches/
    │   └── labels.csv
    └── ValSet/
        ├── original_patches/
        ├── warped_patches/
        └── labels.csv
```

The generated TrainSet and ValSet folders contain:
- original_patches: Original image patches
- warped_patches: Corresponding warped versions
- labels.csv: Contains corner coordinates and image IDs

The generated data will be used for both supervised and unsupervised training approaches in the project.

### 2. Supervised Learning Approach
Implementation of a supervised deep learning model that:
- Directly regresses homography parameters from image pairs
- Uses ground truth homography matrices for training
- Learns feature detection and matching in an end-to-end manner

### 3. Unsupervised Learning Approach
Development of an unsupervised learning framework that:
- Learns homography estimation without explicit ground truth
- Utilizes photometric consistency as a supervision signal
- Implements a differentiable spatial transformer for image warping

This project aims to demonstrate the advantages of deep learning in panorama stitching, offering improved robustness and efficiency compared to traditional methods, while maintaining high generalization capabilities across diverse visual scenarios.


I'll add a section about running the training code to the README:

## Training

### Running the Training Script
```bash
python Train.py [arguments]
```

### Arguments
- `--BasePath`: Base path for data directory (default: "../Data")
- `--CheckPointPath`: Path to save checkpoints (default: "../Checkpoints/")
- `--ModelType`: Choose between supervised ("Sup") or unsupervised ("Unsup") training (default: "Unsup")
- `--NumEpochs`: Number of epochs to train (default: 50)
- `--DivTrain`: Factor to reduce training data per epoch (default: 1)
- `--MiniBatchSize`: Size of mini-batches (default: 32)
- `--LoadCheckPoint`: Load model from latest checkpoint (0/1) (default: 0)
- `--LogsPath`: Path to save tensorboard logs (default: "Logs/")

### Training Process
The training script:
- Supports both supervised and unsupervised approaches
- Uses Adam optimizer with learning rate 0.0005
- Implements learning rate scheduling
- Saves model checkpoints periodically
- Logs training and validation losses using wandb and tensorboard
- Performs validation after each epoch
- Uses GPU if available, otherwise CPU

I'll add this section to the README to explain how to run both supervised and unsupervised training with their respective checkpoint directories:

### Running the Training

#### Supervised Learning
```bash
python Train.py --ModelType Sup --NumEpochs 50 --MiniBatchSize 32 --CheckPointPath ../Checkpoints_Sup/
```
This will:
- Train the model using the supervised approach
- Run for 50 epochs with batch size of 32
- Create a `Checkpoints_Sup` directory in Phase2 folder
- Save all model checkpoints in `Checkpoints_Sup`

```bash
Phase2/
└── Checkpoints_Sup/
    └── [supervised model checkpoints]
```

#### Unsupervised Learning
```bash
python Train.py --ModelType UnSup --NumEpochs 50 --MiniBatchSize 32 --CheckPointPath ../Checkpoints_UnSup/
```
This will:
- Train the model using the unsupervised approach
- Run for 50 epochs with batch size of 32
- Create a `Checkpoints_UnSup` directory in Phase2 folder
- Save all model checkpoints in `Checkpoints_UnSup`
```bash
Phase2/
└── Checkpoints_UnSup/
    └── [unsupervised model checkpoints]
```

The checkpoints saved during training can be used later for testing or continuing training from a specific checkpoint.


### Running the Wrapper Script

The Wrapper.py script performs panorama stitching using either supervised or unsupervised models.

#### For Unsupervised Model
For the model uploaded on Onedrive

```bash
python Wrapper.py --path ../Data/P1Ph2TestSet/Phase2Pano/tower/ --ModelType UnSup --CheckPointPath ../Checkpoints/unsup_model.ckpt
```
For any new Trained Model
```bash
python Wrapper.py --path ../Data/P1Ph2TestSet/Phase2Pano/tower/ --ModelType UnSup --CheckPointPath ../Checkpoints_UnSup/0a0model.ckpt
```

#### For Supervised Model

For the model uploaded on Onedrive

```bash
python Wrapper.py --path ../Data/P1Ph2TestSet/Phase2Pano/tower/ --ModelType Sup --CheckPointPath ../Checkpoints/sup_model.ckpt
```
For any new Trained Model
```bash
python Wrapper.py --path ../Data/P1Ph2TestSet/Phase2Pano/tower/ --ModelType Sup --CheckPointPath ../Checkpoints_Sup/0a0model.ckpt
```

### Arguments
- `--path`: Path to the directory containing images to be stitched
- `--ModelType`: Choose between supervised ("Sup") or unsupervised ("UnSup") model
- `--CheckPointPath`: Path to the model checkpoint file

Based on the code, here's the relevant tree structure showing the image source, results, and checkpoints locations:

```
RBE549-AutoPano/
└── Phase2/
    ├── Code/
    │   └── Wrapper.py
    ├── Data/
    │   └── P1Ph2TestSet/
    │       ├── Phase2Pano/
    │       │   └── tower/
    │       │       └── [input images]
    │       └── Results/
    │           └── tower/
    │               └── mypanoN.jpg
    ├── Checkpoints_Sup/
    │   └── [supervised model checkpoints]
    └── Checkpoints_UnSup/
        └── [unsupervised model checkpoints]
```

This structure shows:
- Input images are read from `Phase2/Data/P1Ph2TestSet/Phase2Pano/tower/`
- Results are saved in `Phase2/Data/P1Ph2TestSet/Results/tower/`
- Supervised model checkpoints are in `Phase2/Checkpoints_Sup/`
- Unsupervised model checkpoints are in `Phase2/Checkpoints_UnSup/`

The script will:
1. Load the specified model (supervised or unsupervised).
2. Read images from the input directory.
3. Perform sequential image stitching.
4. Save intermediate results in the Results directory.
5. Final panorama will be saved as 'mypano{i}.jpg' where i is the number of images stitched.



## Testing

### Data Generation
The testing pipeline first generates test patches from input images:
- Creates patches of size 128x128 from test images
- Stores original and warped patches
- Generates corresponding labels
- Saves data in appropriate directories

### Running the Testing Code
To test the trained models, use the following commands:

#### For Supervised Model

For the model uploaded on Onedrive

```bash
python Test.py --ModelType Sup --ModelPath ../Checkpoints/sup_model.ckpt --BasePath ../Data/P1Ph2TestSet/
```
For any new Trained Model
```bash
python Test.py --ModelType Sup --ModelPath ../Checkpoints_Sup/0a0model.ckpt --BasePath ../Data/P1Ph2TestSet/
```

#### For Unsupervised Model
For the model uploaded on Onedrive

```bash
python Test.py --ModelType UnSup --ModelPath ../Checkpoints/unsup_model.ckpt --BasePath ../Data/P1Ph2TestSet/
```
For any new Trained Model
```bash
python Test.py --ModelType UnSup --ModelPath ../Checkpoints_UnSup/0a0model.ckpt --BasePath ../Data/P1Ph2TestSet/
```

### Arguments
- `--ModelPath`: Path to the trained model checkpoint
- `--BasePath`: Path to test data directory
- `--ModelType`: Choose between supervised ("Sup") or unsupervised ("UnSup") model
- `--LabelsPath`: Path to labels file (optional)

### Directory Structure
```
Phase2/
├── Data/
│   └── P1Ph2TestSet/
│       ├── Phase2/
│       │   └── [test images]
│       ├── TestSet/
│       │   ├── original_patches/
│       │   ├── warped_patches/
│       │   └── labels.csv
│       └── Results/
│           └── [test results]
```

The testing script:
- Loads the specified model checkpoint
- Generates test patches from input images
- Computes EPE (End Point Error) and testing loss
- Logs results using wandb
- Displays average EPE and testing loss for the entire test set


For Downloading saved Checkpoints, use the below link.

https://wpi0-my.sharepoint.com/:f:/g/personal/nbahadarpurkar_wpi_edu/Eo0kn391lGZNrHtasrLI3hkBXjImlzYJw_D-GGKBeR46HQ?e=g6u0Nr




