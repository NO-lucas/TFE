#!/bin/bash

#SBATCH --job-name=cyto_fine_tune  # Nom du job
#SBATCH --output=cyto_fine_tune_classifier.out  # Fichier de sortie
#SBATCH --error=cyto_fine_tune_classifier.err  # Fichier d'erreur

#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --qos=preemptible

# Charger les modules nécessaires

# Activation de l'environnement virtuel
source ~/cyto_env/bin/activate
module load cp3
module load releases/2021a
module load Python/3.9.5-GCCcore-10.3.0
# module load python/python36_sl7_gcc73
module load releases/2023a
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load matplotlib/3.7.2-gfbf-2023a

# Itération sur les datasets et les valeurs de rank
for dataset in "cyto"; do
    for rank in 8; do
        for shot in 600; do
            echo "Lancement avec dataset = $dataset et rank = $rank"
            python ~/Cytology-fine-tuning/run.py \
                --seed_launch "0" \
                --shots_launch "$shot" \
                --lr_launch "5e-4" \
                --iterations 200 \
                --rank_launch "$rank" \
                --model_launch "clip" \
                --dataset_launch "$dataset" \
                --task_launch image_classifier
            echo "Fin de l'exécution pour dataset = $dataset et rank = $rank"
        done
    done
done
