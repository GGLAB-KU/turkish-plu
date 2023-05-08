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

python3 filter_next_event_prediction.py \
  --tokenizer_path dbmdz/bert-base-turkish-uncased \
  --simcse_path ardauzunoglu/sup-simcse-tr-bert-base \
  --next_event_prediction_path data/next-event-prediction.csv \
  --step2goal data/merged-wikihow-tr-step2goal.json \
  --wikihow_path data/merged-wikihow-tr-wikihow.json \
  --save_path filtered_next_event_prediction.json \