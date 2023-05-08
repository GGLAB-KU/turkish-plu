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

python3 rerank/train.py \
  --train_null \
  --add_goal \
  --use_para_score \
  --model_name distilbert-base \
  --context_length 1 \
  --step_goal_file data/merged-wikihow-tr-wikihow.json \
  --step_goal_map_file data/merged-wikihow-tr-step2goal.json \
  --train_file data/gold.rerank.org.t30.train.json \
  --dev_file data/gold.rerank.org.t30.test.json \
  --gold_step_goal_para_score data/gold.para.base.all.score \
  --save_path model/1.00-original-0.00-translated-data/distilberturk.epoch.reranker.pt \
  --neg_num 29 --bs 1 \
  --mega_bs 4 --val_bs 1 \
  --min_save_ep 0 \
  --epochs 5