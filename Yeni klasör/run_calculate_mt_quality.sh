#!/bin/bash
#SBATCH --job-name=Reproduce_SimCSE            
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1    
#SBATCH --partition=ai  
#SBATCH --qos=ai  
#SBATCH --account=ai  
#SBATCH --gres=gpu:tesla_t4:1  
#SBATCH --time=60:0:0        
#SBATCH --output=test-%j.out    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=foo@bar.com
#SBATCH --mem=60GB

python3 calculate_mt_quality.py