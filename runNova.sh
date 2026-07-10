#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nova
#SBATCH --job-name="sciml"
#SBATCH --mail-user=username@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --output="sciml_%j.out"
#SBATCH --error="sciml_%j.err"

module load intel

source /work/mech-ai/username/packages/geometry_matters/sciml/bin/activate

python main.py --config configs/compositional/conf.yaml


