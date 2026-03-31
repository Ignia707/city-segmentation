# City Street Semantic Segmentation

Real-time semantic segmentation of urban street scenes.
Ideation and implementation of a different approachs.

## Team Approaches

| Member | Approach | Best Val mIoU |
|---|---|---|
| Ignia | Modified DeepLabV3+ with MobileNetV2 + FEM attention | 50.49% (baseline) |
| Varun | HRnet V2 - W48 + OCR | 68.7% |
| Subrajith | TBD |  |
| Raghunandan | SegFormer-B0 (Cityscapes, pretrained, 20 samples) | 54.44%   |
|            | SegFormer-B3 (Cityscapes, pretrained)             | 62.85%   |

## Repo Structure
```
models/
├── ignia/          — DeepLabV3+ with custom attention
├── Varun/          — TBD
├── Subrajith/      — TBD
└── Raghunandan/    — TBD
```
## Dataset
We use the [Cityscapes Dataset](https://www.cityscapes-dataset.com/) for all experiments.

### Download
1. Register at https://www.cityscapes-dataset.com/register/
2. Download these two files:
   - `gtFine_trainvaltest.zip` (241MB) — annotations
   - `leftImg8bit_trainvaltest.zip` (11GB) — images
3. Place and extract both under `data/cityscapes/` at the repo root


## Setup
Refer individual README-files in respective directories
