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
#SBATCH --job-name=train_image_classification
#SBATCH --output=/home2/faculty/wjakubowski/logs/cnn/cnn_3.log

. /home2/faculty/wjakubowski/miniconda3/etc/profile.d/conda.sh
conda activate cnn

for (( state=101; state<=506; state+=101 )); do
    python /mnt/evafs/groups/ganzha_23/wjakubowski/ConvolutionalNeuralNetworks/src/models/cnn.py \
        --data "/mnt/evafs/groups/ganzha_23/wjakubowski/ConvolutionalNeuralNetworks/data/" \
        --outputdir "/mnt/evafs/groups/ganzha_23/wjakubowski/ConvolutionalNeuralNetworks/src/models" \
        --model "cnn3" \
        --seed "${state}"
done