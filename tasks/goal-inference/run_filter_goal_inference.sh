#!/bin/bash
#SBATCH --job-name=Reproduce_SimCSE            
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=4    
#SBATCH --partition=ai        
#SBATCH --qos=ai        
#SBATCH --account=ai        
#SBATCH --gres=gpu:tesla_t4:1    
#SBATCH --time=96:0:0        
#SBATCH --output=test-%j.out    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=foo@bar.com
#SBATCH --mem=60GB

python3 filter_goal_inference.py \
  --tokenizer_path dbmdz/bert-base-turkish-uncased \
  --simcse_path ardauzunoglu/sup-simcse-tr-bert-base \
  --goal_inference_path translated_goal_inference.json \
  --step2goal data/translated-wikihow-tr-step2goal.json \
  --wikihow_path data/translated-wikihow-tr-wikihow.json \
  --save_path filtered_translated_goal_inference.json \