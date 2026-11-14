#!/bin/bash
# VideoMAE V2 ViT-S pre-training on SSv2, 10 epochs, 1x L40S (48GB).
# Submit from the repo root:  sbatch scripts/pretrain/vit_s_ssv2_pt_10ep_1xl40s.sbatch
#
# Batch size: on V100 16GB, batch 8 x 2 views uses ~60% (~9.6GB, ~1-1.2GB per batch unit).
# L40S 48GB at ~75-85% -> 34-39 per GPU; we use 32 to keep headroom.
# NOTE: needs a PyTorch 2.x / CUDA >= 11.8 env; the cu113 V100 env cannot run on L40S (sm_89).
#SBATCH --job-name=vit_s_ssv2_pt_10ep_1xl40s
#SBATCH --partition=gl40s_short             # <-- your L40S partition (check: sinfo -o "%P %l %G")
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1            # <-- GPU type string / count as sinfo shows it
#SBATCH --cpus-per-task=16           # 15 video-decoding workers for the one GPU
#SBATCH --mem=64G
#SBATCH --time=12:00:00              # ~6-8 h expected for 10 epochs on 1x L40S; must fit the partition limit
#SBATCH --output=%x-%j.out

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate /gpfs/scratch/sd6701/conda-envs/videomae-l40s

cd "${SLURM_SUBMIT_DIR}"

DATA=/gpfs/scratch/sd6701/personal/salt/datasets/ssv2/pretrain_train.txt
OUT=/gpfs/scratch/sd6701/personal/salt/work_dir/vit_s_ssv2_pt_10ep_1xl40s
mkdir -p "${OUT}"

# use however many GPUs Slurm actually gave us
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
WORKERS=$(( SLURM_CPUS_PER_TASK / NGPU - 1 ))
echo "GPUs: ${NGPU}, data workers per GPU: ${WORKERS}"

export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# export WANDB_MODE=offline         # uncomment if compute nodes have no internet

# --auto_resume is on by default: resubmitting continues from the last
# checkpoint-<epoch>.pth in ${OUT}. Keep the same GPU count when resuming.
# --save_ckpt_freq 1 -> checkpoint-0.pth ... checkpoint-9.pth, one after every epoch.
torchrun --standalone --nproc_per_node="${NGPU}" run_mae_pretraining.py \
  --data_path "${DATA}" \
  --model pretrain_videomae_small_patch16_224 \
  --decoder_depth 4 \
  --mask_type tube --mask_ratio 0.9 \
  --decoder_mask_type run_cell --decoder_mask_ratio 0.5 \
  --num_frames 16 --sampling_rate 2 \
  --batch_size 32 --num_sample 2 \
  --num_workers "${WORKERS}" \
  --opt adamw --opt_betas 0.9 0.95 --lr 1e-3 \
  --epochs 10 --warmup_epochs 1 --save_ckpt_freq 1 \
  --output_dir "${OUT}" \
  --log_dir "${OUT}" \
  --wandb_project videomaev2 --wandb_run_name "vit_s_ssv2_pt_10ep_1xl40s_${SLURM_JOB_ID}" \
  --wandb_tags ssv2 vit_s pretrain 1xl40s
