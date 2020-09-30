#!/bin/bash

#SBATCH --mail-user=liao@tnt.uni-hannover.de
#SBATCH --mail-type=ALL             # Eine Mail wird bei Job-Start/Ende versendet
#SBATCH --job-name=paralt2       # Name unter dem der Job in der Job-History gespeichert wird
#SBATCH --output=slurm_log/rel_paral_type2.txt         # Logdatei für den merged STDOUT/STDERR output
#SBATCH --partition=gpu_cluster_enife # Partition auf der gerechnet werden soll
                                    #   ohne Angabe des Parameters wird auf der Default-Partition gerechnet
#SBATCH --nodes=1                   # Reservierung von 2 Rechenknoten
                                    #   alle nachfolgend reservierten CPUs müssen sich auf den reservierten Knoten befinden
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
srun hostname
echo "Here begin to  rel_paral_type2 "
working_dir=~/work_code/AoANet
cd $working_dir
source activate aoa
# sh train_refine.sh
# sh train_v3d1.sh
python train_rel.py --fuse_type 2
# python eval_ensemble.py
echo "training  rel_paral_type2!"
