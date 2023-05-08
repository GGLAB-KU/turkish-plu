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

python3 rerank/test_rerank.py \
    --retrieve_model_name_or_path ardauzunoglu/sup-simcse-tr-bert-base \
    --model_path ./model/1.00-original-0.00-translated-data/distilberturk.ep4.reranker.pt \
    --raw_test_path data/merged-hyperlinks-test.json \
    --test_path retrieved-merged-hyperlinks-test.json \
    --titles_file data/merged-wikihow-tr-goals.txt \
    --save_path reranked-step-goal-matches-test.json \
    --no_label \