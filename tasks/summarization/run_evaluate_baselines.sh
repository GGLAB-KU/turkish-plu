#!/bin/bash
#SBATCH --job-name=Reproduce_SimCSE            
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=4    
#SBATCH --partition=ai        
#SBATCH --qos=ai        
#SBATCH --account=ai        
#SBATCH --gres=gpu:tesla_t4:4    
#SBATCH --time=36:0:0        
#SBATCH --output=test-%j.out    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=foo@bar.com
#SBATCH --mem=240GB

NUM_GPU=2
PORT_ID=$(expr $RANDOM + 1000)
export OMP_NUM_THREADS=2

python3 -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID evaluate_baselines.py \
    --model_name mukayese/transformer-turkish-summarization \
