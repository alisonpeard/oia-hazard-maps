#!/bin/bash
#SBATCH --job-name=hazmaps
#SBATCH --output=logs/sm.out
#SBATCH --error=logs/sm.err
#SBATCH --partition=Short
#SBATCH --time=08:00:00

# python annual_stats.py # 
python scripts/hazard_model_detrended.py
# python hazard_ensemble.py