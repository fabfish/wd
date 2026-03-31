# Rebuttal Experiments

## Reviewer qZ4a (RvkYepRfn2) — Score: 2 (Reject)

### Experiment-Related Criticisms

1. **No multi-seed averaging**: "Experiments are not averaged over several random seeds to remove the effect of the noise."
2. **Insufficient configurations**: "The authors should add more training configurations and average over several runs to be able to say that the theory is predictive."
3. **Small number of runs**: "Empirical claims regarding the interplay between η, λ, B in practice are based on small number of runs. Therefore, I do not find them convincing."

### Response Plan

| Phase | Model | Seed | Experiments | Runs | Est. Time | Status |
|-------|-------|------|-------------|------|-----------|--------|
| Baseline (done) | ResNet-18 | 42 | Exp 1/2/3 | 83 | — | ✅ |
| Phase 1 | ResNet-18 | 123 | Exp 1/2/3 | 83 | ~5.5h | ✅ |
| Phase 2 | VGG-16 | 42 | Exp 1/2/3 | 83 | ~5.5h | ✅ |
| Phase 3 | VGG-16 | 123 | Exp 1/2/3 | 83 | ~5.5h | ✅ |
| Phase 4 | ResNet-50 | 42 | Exp 1/2/3 | 83 | ~9h | ✅ |
| Phase 5 | ResNet-50 | 123 | Exp 1/2/3 | 83 | ~9h | ✅ |

**Hardware**: 3x NVIDIA A6000 Pro (48GB) for ResNet-18; 8x GPU (49GB) with 4 workers/GPU for ResNet-50

### Experiment Details

- **Exp 1** (24 runs): Optimal LR ordering — SGD vs SGD+WD vs SGDM+WD
  - `batch_size=128`, `lr ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0}`
- **Exp 2** (35 runs): η-λ interaction heatmap (SGDM)
  - `lr ∈ {0.01, 0.05, 0.1, 0.2, 0.3}`, `wd ∈ {1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2}`
- **Exp 3** (24 runs): Batch size scaling with linear LR rule
  - `batch_size ∈ {64, 128, 256, 512}`, `wd ∈ {1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3}`

### Results

Results are saved in `rebuttal/results/` with naming convention:
- `results_{model}_seed{seed}.csv`

### Reports

| Report | Content |
|--------|---------|
| [`phase1_report.md`](phase1_report.md) | ResNet-18 2-seed reproducibility (seed=42 vs 123) |
| [`phase2_3_report.md`](phase2_3_report.md) | VGG-16 cross-architecture validation (2 seeds) |
| [`resnet18_4run_report.md`](resnet18_4run_report.md) | ResNet-18 combined 4-run report (N=4) |
| [`resnet50_report.md`](resnet50_report.md) | ResNet-50 cross-architecture validation (2 seeds) |

### Usage

```bash
# Phase 1: ResNet-18 with new seed
python rebuttal/run_rebuttal.py --model resnet18 --seed 123 --experiment 1 --gpus 0,1,2
python rebuttal/run_rebuttal.py --model resnet18 --seed 123 --experiment 2 --gpus 0,1,2
python rebuttal/run_rebuttal.py --model resnet18 --seed 123 --experiment 3 --gpus 0,1,2

# Phase 2: VGG-16
python rebuttal/run_rebuttal.py --model vgg16 --seed 42 --experiment 1 --gpus 0,1,2
python rebuttal/run_rebuttal.py --model vgg16 --seed 42 --experiment 2 --gpus 0,1,2
python rebuttal/run_rebuttal.py --model vgg16 --seed 42 --experiment 3 --gpus 0,1,2

# Phase 4-5: ResNet-50 (8-GPU, 4 workers/GPU)
nohup bash -c 'for SEED in 42 123; do for EXP in 1 2 3; do \
  python3 rebuttal/run_rebuttal.py --model resnet50 --experiment $EXP \
    --seed $SEED --gpus all --epochs 100 --workers_per_gpu 4; \
done; done' > rebuttal/resnet50.log 2>&1 &
```
