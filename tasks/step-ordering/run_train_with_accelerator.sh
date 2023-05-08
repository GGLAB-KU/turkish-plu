#!/bin/bash
#SBATCH --job-name=Reproduce_SimCSE            
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=4    
#SBATCH --partition=ai        
#SBATCH --qos=ai        
#SBATCH --account=ai        
#SBATCH --gres=gpu:tesla_t4:4 
#SBATCH --time=12:0:0        
#SBATCH --output=test-%j.out    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=foo@bar.com
#SBATCH --mem=240GB

accelerate config

accelerate test

accelerate launch train_with_accelerator.py \
  --model_name_or_path xlm-roberta-base \
  --train_file data/step-ordering-train.csv \
  --validation_file data/step-ordering-val.csv \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --output_dir model/xlm-r-step-ordering-model-bs16-4/ \