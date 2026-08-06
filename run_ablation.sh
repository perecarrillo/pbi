#!/usr/bin/env bash

N_REPS=${1:-5}
OUTPUT_DIR=${2:-ablation_output_val}

micromamba activate pbi-dnabert

python ablation_study.py \
  --base_config model_configs/predphi_optimized.yaml \
  --experiment embeddings_phage \
  --n_reps $N_REPS \
  --output_dir $OUTPUT_DIR/embeddings_phage

python ablation_study.py \
  --base_config model_configs/predphi_optimized.yaml \
  --experiment embeddings_bacteria \
  --n_reps $N_REPS \
  --output_dir $OUTPUT_DIR/embeddings_bacteria

python ablation_study.py \
  --base_config model_configs/predphi_optimized.yaml \
  --experiment kmer \
  --n_reps $N_REPS \
  --output_dir $OUTPUT_DIR/kmer

python ablation_study.py \
  --base_config model_configs/predphi_optimized.yaml \
  --experiment reducer \
  --n_reps $N_REPS \
  --output_dir $OUTPUT_DIR/reducer

python ablation_study.py \
  --base_config model_configs/predphi_optimized.yaml \
  --experiment ensemble \
  --n_reps $N_REPS \
  --output_dir $OUTPUT_DIR/ensemble