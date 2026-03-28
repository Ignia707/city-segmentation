# Ignia — Modified DeepLabV3+ with FEM Attention

## Approach
Modified DeepLabV3+ with MobileNetV2 backbone and custom
feature fusion with FEM attention mechanism.

## Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA support first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install remaining dependencies
pip install -r requirements.txt
```

## Training
```bash
cd models/ignia
python -m src.train
```

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