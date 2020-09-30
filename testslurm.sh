#!/bin/bash

#SBATCH --mail-user=<UserName>@tnt.uni-hannover.de
#SBATCH --mail-type=ALL             # Eine Mail wird bei Job-Start/Ende versendet
#SBATCH --job-name=MyTestjob        # Name unter dem der Job in der Job-History gespeichert wird
#SBATCH --output=result.txt         # Logdatei für den merged STDOUT/STDERR output

#SBATCH --time=01:30:00             # Maximale Laufzeit des Jobs, bis Slurm diesen abbricht (HH:MM:SS)
#SBATCH --partition=gpu_cluster_pascal     # Partition auf der gerechnet werden soll
                                    #   ohne Angabe des Parameters wird auf der Default-Partition gerechnet
#SBATCH --nodes=2                   # Reservierung von 2 Rechenknoten
                                    #   alle nachfolgend reservierten CPUs müssen sich auf den reservierten Knoten befinden
#SBATCH --tasks-per-node=2          # Reservierung von 2 CPUs pro Rechernknoten
#SBATCH --mem-per-cpu=100M          # Reservierung von 400MB RAM Speicher (100MB pro verwendeter CPU)

echo "Hier beginnt die Ausführung/Berechnung"
working_dir=~
cd $working_dir
srun hostname
srun sleep 60
