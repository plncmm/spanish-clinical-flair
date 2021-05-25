#!/bin/bash
#SBATCH -J backward-flair-bio
#SBATCH -p gpus
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -o backward-flair-bio_%j.out
#SBATCH -e backward-flair-bio_%j.err
#SBATCH --mail-user=villenafabian@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --mem-per-cpu=4300
#SBATCH --gres=gpu:1

ml Python/3.7.3
ml CUDA/10.2.89

/home/fvillena/bio-flair/venv/bin/python /home/fvillena/bio-flair/bio_flair.py train backward