#!/bin/bash
#SBATCH --job-name=Reproduce_SimCSE            
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=4    
#SBATCH --partition=mid        
#SBATCH --qos=users        
#SBATCH --account=users    
#SBATCH --gres=gpu:tesla_t4:1    
#SBATCH --time=12:0:0        
#SBATCH --output=test-%j.out    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=foo@bar.com
#SBATCH --mem=50GB

python3 retrieve.py \
    --model_name_or_path ardauzunoglu/sup-simcse-tr-bert-base \
    --retrieve_for train \
    --gold_matches_train data/original-wikihow-tr-hyperlinks-train.json \
    --gold_matches_test data/merged-hyperlinks-test.json \
    --titles_file data/merged-wikihow-tr-goals.txt \
    --steps_file data/merged-wikihow-tr-steps.txt \
    --step2goal data/merged-wikihow-tr-step2goal.json \
