#!/bin/bash

#SBATCH --mail-user=liao@tnt.uni-hannover.de
#SBATCH --mail-type=ALL             # Eine Mail wird bei Job-Start/Ende versendet
#SBATCH --job-name=3layer # Name unter dem der Job in der Job-History gespeichert wird
#SBATCH --output=./slurm_log/v3d1_3layer.txt         # Logdatei für den merged STDOUT/STDERR output
#SBATCH --partition=gpu_cluster_enife # Partition auf der gerechnet werden soll
                                    #   ohne Angabe des Parameters wird auf der Default-Partition gerechnet
#SBATCH --time=240:59:59
#SBATCH --nodes=1                   # Reservierung von 2 Rechenknoten
                                    #   alle nachfolgend reservierten CPUs müssen sich auf den reservierten Knoten befinden
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
echo "Here begin to v3d1_3layer"
working_dir=~/work_code/AoANet
cd $working_dir
srun hostname
source activate aoa
# sh train_refine.sh
# sh train.sh
# sh train_v3d1_warmup.sh
sh train_v3d1.sh
# python eval.py
# python eval_ensemble_online.py
echo "training  v3d1_3layer!"
