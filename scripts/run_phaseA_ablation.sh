#!/usr/bin/env bash
set -euo pipefail

# Phase-A ablation:
#   1) lr pair (phase1/phase2)
#   2) phase1 switch epoch
#   3) weight decay
#   4) batch size / input size
#
# Usage:
#   bash scripts/run_phaseA_ablation.sh
#
# Optional env:
#   CUDA_VISIBLE_DEVICES=0
#   EPOCHS_TOTAL=500
#   TEST_FREQ=30
#   NUM_WORKERS=8

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS_TOTAL="${EPOCHS_TOTAL:-500}"
TEST_FREQ="${TEST_FREQ:-30}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MODEL_NAME="${MODEL_NAME:-mobile_sam_adapter}"
CKPT_PATH="${CKPT_PATH:-./checkpoints/mobile_sam.pt}"
DATASET_PATH="${DATASET_PATH:-./data/orange}"
GT_FOLDER="${GT_FOLDER:-./data/orange/masks/}"
TEST_LIST_FILE="${TEST_LIST_FILE:-./data/orange/imageset/test.txt}"

run_case() {
  local tag="$1"
  local p1_lr="$2"
  local p2_lr="$3"
  local p1_ep="$4"
  local wd="$5"
  local bs="$6"
  local inp="$7"

  echo "============================================================"
  echo "[RUN] ${tag}"
  echo "PHASE1_LR=${p1_lr}, PHASE2_LR=${p2_lr}, PHASE1_EPOCHS=${p1_ep}, WEIGHT_DECAY=${wd}, BATCH_SIZE=${bs}, INP_SIZE=${inp}"
  echo "============================================================"

  PHASE1_LR="${p1_lr}" \
  PHASE2_LR="${p2_lr}" \
  PHASE1_EPOCHS="${p1_ep}" \
  WEIGHT_DECAY="${wd}" \
  BATCH_SIZE="${bs}" \
  INP_SIZE="${inp}" \
  EPOCHS_TOTAL="${EPOCHS_TOTAL}" \
  TEST_FREQ="${TEST_FREQ}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  MODEL_NAME="${MODEL_NAME}" \
  CKPT_PATH="${CKPT_PATH}" \
  DATASET_PATH="${DATASET_PATH}" \
  GT_FOLDER="${GT_FOLDER}" \
  TEST_LIST_FILE="${TEST_LIST_FILE}" \
  EXP_NAME="ablation_A_${tag}" \
  "${PYTHON_BIN}" train_ablation.py
}

## Baseline
#run_case "baseline"           "1e-3"  "1e-4"  "200" "1e-2" "4" "1024"
#
## A1: LR pair
#run_case "lr_5e4_5e5"         "5e-4"  "5e-5"  "200" "1e-2" "4" "1024"
#run_case "lr_1e3_5e5"         "1e-3"  "5e-5"  "200" "1e-2" "4" "1024"
#run_case "lr_5e4_1e4"         "5e-4"  "1e-4"  "200" "1e-2" "4" "1024"
#
## A2: Switch epoch
#run_case "switch_100"         "1e-3"  "1e-4"  "100" "1e-2" "4" "1024"
#run_case "switch_300"         "1e-3"  "1e-4"  "300" "1e-2" "4" "1024"
#
## A3: Weight decay
#run_case "wd_1e3"             "1e-3"  "1e-4"  "200" "1e-3" "4" "1024"
#run_case "wd_5e2"             "1e-3"  "1e-4"  "200" "5e-2" "4" "1024"

# A4: batch size / input size tradeoff
run_case "bs2_inp384"         "1e-3"  "1e-4"  "200" "1e-2" "2" "384"
run_case "bs1_inp512"         "1e-3"  "1e-4"  "200" "1e-2" "1" "512"

echo "All Phase-A ablation runs have finished."