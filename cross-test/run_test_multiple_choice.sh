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
#SBATCH --mem=60GB

python3 test_multiple_choice.py \
  --model_name_or_path model/berturk-next-event-prediction-model-bs16 \
  --tokenizer_name dbmdz/bert-base-turkish-cased \
  --per_device_test_batch_size 8 \
  --test_file data/step-inference-test.csv \