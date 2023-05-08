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

python3 classify_orderliness.py \
  --model_name ardauzunoglu/berturk-base-for-step-ordering-classification \
  --data data/merged-wikihow-tr-wikihow.json \
  --output_file data/merged-wikihow-tr-wikihow.json \