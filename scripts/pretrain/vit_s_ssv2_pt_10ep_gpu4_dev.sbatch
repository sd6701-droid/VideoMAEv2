#!/bin/bash
# VideoMAE V2 ViT-S pre-training on SSv2, 10 epochs, 4 GPUs (gpu4_dev), single node.
# Submit from the repo root:  sbatch scripts/pretrain/vit_s_ssv2_pt_10ep_gpu4_dev.sbatch
#SBATCH --job-name=vit_s_ssv2_pt_10ep
#SBATCH --partition=gpu4_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4                # add a type if required, e.g. gpu:v100:4
#SBATCH --cpus-per-task=40           # ~10 video-decoding workers per GPU
#SBATCH --mem=160G
#SBATCH --time=04:00:00              # partition max (check: sinfo -p gpu4_dev -o "%P %l %G %D")
#SBATCH --output=%x-%j.out

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate /gpfs/scratch/sd6701/conda-envs/videomae

cd "${SLURM_SUBMIT_DIR}"

DATA=/gpfs/scratch/sd6701/personal/salt/datasets/ssv2/pretrain_train.txt
OUT=/gpfs/scratch/sd6701/personal/salt/work_dir/vit_s_ssv2_pt_10ep_4gpu
mkdir -p "${OUT}"

export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# export WANDB_MODE=offline         # uncomment if compute nodes have no internet

# --auto_resume is on by default: resubmitting this script continues from the
# last checkpoint-<epoch>.pth in ${OUT} (e.g. after hitting the time limit).
torchrun --standalone --nproc_per_node=4 run_mae_pretraining.py \
  --data_path "${DATA}" \
  --model pretrain_videomae_small_patch16_224 \
  --decoder_depth 4 \
  --mask_type tube --mask_ratio 0.9 \
  --decoder_mask_type run_cell --decoder_mask_ratio 0.5 \
  --num_frames 16 --sampling_rate 2 \
  --batch_size 8 --num_sample 2 \
  --num_workers 10 \
  --opt adamw --opt_betas 0.9 0.95 --lr 1e-3 \
  --epochs 10 --warmup_epochs 1 --save_ckpt_freq 1 \
  --output_dir "${OUT}" \
  --log_dir "${OUT}" \
  --wandb_project videomaev2 --wandb_run_name "vit_s_ssv2_pt_10ep_gpu4_${SLURM_JOB_ID}" \
  --wandb_tags ssv2 vit_s pretrain gpu4_dev
