#!/bin/bash
#SBATCH -p cs -q cspg
#SBATCH -c4 --mem=15g
#SBATCH --gres gpu:1
module load nvidia/cuda-11.7

source /home/hfyyk1/env/aberration/bin/activate
source /usr2/share/gpu.sbatch        


python train_eval.py --config train.yaml

