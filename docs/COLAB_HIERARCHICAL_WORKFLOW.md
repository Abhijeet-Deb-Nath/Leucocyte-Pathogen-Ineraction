# Colab Workflow

This project uses Colab for heavy training and evaluation. Local work should stay limited to:

- code changes
- cheap import and smoke checks
- GUI playback of finished checkpoints

## 1. Runtime Setup

Use:

- Python 3
- `T4 GPU` if available

Mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

Clone the repo:

```bash
%cd /content
!rm -rf /content/Leucocyte-Pathogen-Ineraction
!git clone https://github.com/Abhijeet-Deb-Nath/Leucocyte-Pathogen-Ineraction.git
%cd /content/Leucocyte-Pathogen-Ineraction
!python -m pip install -r requirements.txt
!nvidia-smi
!python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

## 2. BC-Only Hierarchical Smoke

```bash
%cd /content/Leucocyte-Pathogen-Ineraction
!mkdir -p /content/runs/hier_bc_smoke
!python -m experiments.train_hierarchical \
  --stages bc \
  --save-dir /content/runs/hier_bc_smoke \
  --seed 42 \
  --bc-pretrain-steps 2048 \
  --bc-epochs 4 \
  --bc-sequence-length 32 \
  --eval-episodes 5 \
  --eval-seed 123 \
  --device cuda
```

Expected artifact:

- `/content/runs/hier_bc_smoke/hierarchical_macrophage_final.pt`

## 3. BC + DAgger Smoke

```bash
%cd /content/Leucocyte-Pathogen-Ineraction
!mkdir -p /content/runs/hier_dagger_smoke
!python -m experiments.train_hierarchical \
  --stages bc dagger \
  --save-dir /content/runs/hier_dagger_smoke \
  --seed 42 \
  --bc-pretrain-steps 4096 \
  --bc-epochs 6 \
  --bc-sequence-length 32 \
  --dagger-iterations 3 \
  --dagger-steps-per-iter 2048 \
  --dagger-epochs-per-iter 3 \
  --eval-episodes 5 \
  --eval-seed 123 \
  --device cuda
```

## 4. BC + DAgger + RL Fine-Tune Smoke

Run this only after the DAgger checkpoint is behaviorally sane.

```bash
%cd /content/Leucocyte-Pathogen-Ineraction
!mkdir -p /content/runs/hier_rl_smoke
!python -m experiments.train_hierarchical \
  --stages bc dagger rl_finetune \
  --save-dir /content/runs/hier_rl_smoke \
  --seed 42 \
  --bc-pretrain-steps 4096 \
  --bc-epochs 6 \
  --bc-sequence-length 32 \
  --dagger-iterations 3 \
  --dagger-steps-per-iter 2048 \
  --dagger-epochs-per-iter 3 \
  --rl-rollout-episodes 8 \
  --rl-iterations 4 \
  --rl-ppo-epochs 2 \
  --eval-episodes 5 \
  --eval-seed 123 \
  --device cuda
```

## 5. Evaluate A Learned Checkpoint

`experiments.evaluate_policies` accepts the hierarchical `.pt` checkpoint through the existing learned-agent interface.

```bash
%cd /content/Leucocyte-Pathogen-Ineraction
!python -m experiments.evaluate_policies \
  --episodes 5 \
  --seed 123 \
  --rl-model /content/runs/hier_dagger_smoke/hierarchical_macrophage_final.pt \
  --obs-mode partial_state \
  --output-csv /content/runs/hier_dagger_smoke_5ep.csv
```

## 6. Save Artifacts To Drive

```bash
!mkdir -p "/content/drive/MyDrive/pathway1_runs/hier_dagger_smoke"
!cp /content/runs/hier_dagger_smoke/hierarchical_macrophage_final.pt "/content/drive/MyDrive/pathway1_runs/hier_dagger_smoke/"
!cp /content/runs/hier_dagger_smoke/hierarchical_training_summary.json "/content/drive/MyDrive/pathway1_runs/hier_dagger_smoke/"
!cp /content/runs/hier_dagger_smoke_5ep.csv "/content/drive/MyDrive/pathway1_runs/hier_dagger_smoke/"
!cp /content/runs/hier_dagger_smoke_5ep_summary.csv "/content/drive/MyDrive/pathway1_runs/hier_dagger_smoke/"
```

## 7. What To Bring Back From Colab

Only bring back:

- the checkpoint
- the training summary JSON
- the evaluation CSV
- the summary CSV

Do not keep large transient run folders in the repo.
