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
#SBATCH --mem=100GB

python3 test_retrieve.py \
    --model_name_or_path ardauzunoglu/sup-simcse-tr-xlm-roberta-base \
    --titles_file data/merged-wikihow-tr-goals.txt \
    --test_data data/merged-hyperlinks-test.json \