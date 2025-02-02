#!/usr/bin/env bash


#SBATCH -A rbe549
#SBATCH -p academic
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -C A30
#SBATCH -t 48:00:00
#SBATCH --mem 64G
#SBATCH --job-name="P1"


# Define log directory inside your project
LOG_DIR=/home/skhobragade/CV/RBE549-AutoPano/Phase2/Code/logs


# Ensure log directory exists
mkdir -p $LOG_DIR

#SBATCH --output=$LOG_DIR/output_train.log
#SBATCH --error=$LOG_DIR/err_train.err


source activate cv
# srun --unbuffered python main.py --train_dqn
python Train.py

