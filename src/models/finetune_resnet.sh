#!/bin/bash
#SBATCH --account=ganzha_23
#SBATCH --partition=short
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=150G
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wiktor.jakubowski.stud@pw.edu.pl
#SBATCH --job-name=finetune_resnet
#SBATCH --output=/home2/faculty/wjakubowski/logs/resnet/resnet_aug.log

. /home2/faculty/wjakubowski/miniconda3/etc/profile.d/conda.sh
conda activate cnn

python /mnt/evafs/groups/ganzha_23/wjakubowski/ConvolutionalNeuralNetworks/src/models/resnet.py \
    --data "/mnt/evafs/groups/ganzha_23/wjakubowski/ConvolutionalNeuralNetworks/data/" \
    --outputdir "/mnt/evafs/groups/ganzha_23/wjakubowski/ConvolutionalNeuralNetworks/src/models"