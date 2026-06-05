#!/bin/bash
#SBATCH --time=16:00:00   
#SBATCH --mem=16g         
#SBATCH -N 1              
#SBATCH -n 4              
#SBATCH -o AgentJob_%j.out     
# #SBATCH -A gk-condo      
#SBATCH -p 3090-gcondo
#SBATCH --gres=gpu:4      

module load anaconda3/2023.09-0-aqbc
source activate agents

module load cuda
nvidia-smi

# Exit on any error
set -e

# Assume that the contract has been generated and approved already.
# Run root solution generation and evolutionary optimization. 
python3 -u main.py --mode root-only --gpu_ids 0 1 2 3
python3 -u main.py --mode evolve-only --gpu_ids 0 1 2 3
