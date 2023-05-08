#!/bin/bash
#SBATCH --job-name=Reproduce_SimCSE            
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=4    
#SBATCH --partition=long        
#SBATCH --qos=users        
#SBATCH --account=users    
#SBATCH --gres=gpu:tesla_t4:1    
#SBATCH --time=96:0:0        
#SBATCH --output=test-%j.out    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=foo@bar.com
#SBATCH --mem=60GB

python3 get_step_embeddings.py \
  --model_path dbmdz/bert-base-turkish-uncased \
  --step2goal data/original-wikihow-tr-step2goal.json \
  --save_path original_step_embeddings.pickle \