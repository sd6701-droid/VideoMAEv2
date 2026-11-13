#!/bin/bash
# VideoMAE V2 ViT-S pre-training on SSv2, 10 epochs, 3x V100 (16GB), single node.
# Submit from the repo root:  sbatch scripts/pretrain/vit_s_ssv2_pt_10ep_3xv100.sbatch
#SBATCH --job-name=vit_s_ssv2_pt_10ep
#SBATCH --partition=gpu8_short             # <-- your V100 partition (check: sinfo -o "%P %l %G")
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:3
#SBATCH --cpus-per-task=30           # ~10 video-decoding workers per GPU
#SBATCH --mem=120G
#SBATCH --time=09:00:00              # ~6.5-7.5 h expected + margin; must fit the partition limit
#SBATCH --output=%x-%j.out

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate /gpfs/scratch/sd6701/conda-envs/videomae

cd "${SLURM_SUBMIT_DIR}"

DATA=/gpfs/scratch/sd6701/personal/salt/datasets/ssv2/pretrain_train.txt
OUT=/gpfs/scratch/sd6701/personal/salt/work_dir/vit_s_ssv2_pt_10ep
mkdir -p "${OUT}"

export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# export WANDB_MODE=offline         # uncomment if compute nodes have no internet

# --auto_resume is on by default: resubmitting this script continues from the
# last checkpoint-<epoch>.pth in ${OUT} (e.g. after hitting the time limit).
torchrun --standalone --nproc_per_node=3 run_mae_pretraining.py \
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
  --wandb_project videomaev2 --wandb_run_name "vit_s_ssv2_pt_10ep_3xv100_${SLURM_JOB_ID}" \
  --wandb_tags ssv2 vit_s pretrain 3xv100
