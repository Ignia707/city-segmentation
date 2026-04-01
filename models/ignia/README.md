# Ignia — Modified DeepLabV3+ with FEM Attention

## Approach
Modified DeepLabV3+ with MobileNetV2 backbone and custom
feature fusion with FEM attention mechanism.

## Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training
```bash
python -m src.train
```

## Results

## Results

| Model | Val mIoU |
|---|---|
| Baseline DeepLabV3+ (MobileNetV2) | 50.49% |

## Status
- [x] Phase 0 — Environment setup
- [x] Phase 1 — Baseline model
- [ ] Phase 2 — Custom modifications
- [ ] Phase 3 — Training
- [ ] Phase 4 — Evaluation
