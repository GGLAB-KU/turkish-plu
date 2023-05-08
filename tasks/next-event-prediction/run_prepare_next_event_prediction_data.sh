#!/bin/bash
#SBATCH --job-name=Reproduce_SimCSE            
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=4    
#SBATCH --partition=ai        
#SBATCH --qos=ai        
#SBATCH --account=ai        
#SBATCH --gres=gpu:tesla_t4:1   
#SBATCH --time=168:0:0        
#SBATCH --output=test-%j.out    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=foo@bar.com
#SBATCH --mem=60GB

python3 prepare_next_event_prediction_data.py \
  --model_path dbmdz/bert-base-turkish-uncased \
  --simcse_path ardauzunoglu/sup-simcse-tr-bert-base \
  --wikihow data/translated-wikihow-tr-wikihow.json \
  --step2goal data/translated-wikihow-tr-step2goal.json \
  --step_embeddings_path translated_step_embeddings.pickle \
  --save_path data/translated-next-event-prediction.csv \