#!/bin/bash
#SBATCH --job-name=Reproduce_SimCSE            
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=4    
#SBATCH --partition=ai  
#SBATCH --qos=ai  
#SBATCH --account=ai  
#SBATCH --gres=gpu:tesla_t4:4  
#SBATCH --time=60:0:0        
#SBATCH --output=test-%j.out    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=foo@bar.com
#SBATCH --mem=240GB

accelerate config

accelerate test

accelerate launch train_with_accelerator.py \
    --model_name_or_path mukayese/mt5-base-turkish-summarization \
    --dataset_name ardauzunoglu/tr-wikihow-summ \
    --num_train_epochs 3 \
    --max_source_length  1024 \
    --max_target_length  128 \
    --num_beams 4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --learning_rate 0.0001 \
    --hub_token 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 \
    --push_to_hub \
    --output_dir models/mt5-base-pro-summ-2