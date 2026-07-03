# Project 1 — CIFAR-10 Image Classification with CNNs

Iterative CNN development on CIFAR-10, progressing from a baseline model through five tuned variants, tracking how each architectural/optimization change affects validation and test accuracy.

## Dataset
- **CIFAR-10**: 60,000 32×32 RGB images, 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- Split: 70% train / 15% val / 15% test, stratified (`random_state=42`)
- Preprocessing: pixel normalization (`/255.0`), one-hot label encoding
- Training: `categorical_crossentropy`, `EarlyStopping` (patience=10, restore best weights), batch size 64, up to 20 epochs

## Results

| Model | Key change vs. previous | Blocks | BatchNorm | Dropout | Augmentation | Optimizer | Val Acc | Test Acc | Val Loss |
|---|---|---|---|---|---|---|---|---|---|
| **Base** | Plain CNN, no regularization | 4 (32/64/128/256) | ✗ | ✗ | ✗ | Adam (0.001) | 0.7163 | 0.7048 | 0.8296 |
| **Model 1** | + BatchNorm after every conv | 4 | ✓ | ✗ | ✗ | Adam (0.001) | 0.7789 | 0.7728 | 0.6821 |
| **Model 2** | BatchNorm →  3 blocks | 3 (32/64/128) | ✗ | ✓ | ✗ | Adam (0.001) | 0.7884 | 0.7869 | 0.6211 |
| **Model 3** | BatchNorm + Dropout + data augmentation | 3 | ✓ | ✓ | ✓ | Adam (0.001) | 0.8249 | **0.8249** | **0.5037** |
| **Model 4** | Same as Model 3 + VGG16 transfer-learning comparison | 3 | ✓ | ✓ | ✓ | Adam (0.001) | 0.8249 | 0.8249 | 0.5037 |
| **Model 5** | Back to 4 blocks, no augmentation, optimizer → SGD | 4 | ✓ | ✓ | ✗ | SGD (0.01) | 0.7860 | 0.7788 | 0.6213 |

**Best model: Model 3 — 82.49% test accuracy, lowest validation loss. Model 4's notebook contains the exact same CNN as Model 3, plus an added VGG16 transfer-learning comparison.

**VGG16 transfer learning (in Model 4 notebook):** 61.24% test accuracy — underperformed the custom CNN. VGG16 is pretrained on 224×224 ImageNet images; using it frozen on 32×32 CIFAR-10 inputs loses detail and limits how well its pretrained features transfer.

## Key findings (the story)
1. **Base → Model 1**: Adding BatchNormalization alone gave the single biggest jump (+7 pts test accuracy) by stabilizing training.
2. **Model 1 → Model 2**: Dropping to 3 conv blocks and using Dropout instead of BatchNorm still improved results slightly — the deeper 4-block base model was likely overcapacity for this data size.
3. **Model 2 → Model 3**: Combining BatchNorm + Dropout + light data augmentation (horizontal flip, small rotation) gave the best result of the set — complementary regularization plus augmentation closed the train/val gap and pushed test accuracy to its peak.
4. **Model 3 → Model 4**: Same architecture; adding a frozen VGG16 feature-extractor as a transfer-learning comparison. VGG16 underperformed the custom CNN (61.2% vs. 82.5%), showing a from-scratch small CNN beats a low-resolution-adapted pretrained backbone here.
5. **Model 4 → Model 5**: Reverting to 4 blocks, dropping augmentation, and switching Adam → SGD *hurt* performance — SGD converged less efficiently within the epoch budget, and the deeper network reintroduced overfitting risk.

**Known issue resolved**: an early normalization bug caused validation accuracy to plateau at random-guess level (~10%); fixed by correcting pixel scaling before the train/val/test split.

## Repo structure
```
Base_Model_Project_1_CNN.ipynb   # Baseline CNN, no regularization
Model_1 Project_1_CNN.ipynb      # + BatchNormalization
Model_2 Project_1_CNN.ipynb      # BatchNorm → Dropout, 3 blocks
Model_3Project_1_CNN.ipynb       # BatchNorm + Dropout + augmentation (best CNN)
Model_4Project_1_CNN.ipynb       # Same CNN as Model 3 + VGG16 comparison
Model_5Project_1_CNN.ipynb       # 4 blocks, no augmentation, SGD optimizer
```

## How to run
1. Open any notebook in Google Colab (GPU runtime recommended)
2. Run cells top to bottom — data loads via `tensorflow.keras.datasets.cifar10`
3. Trained models save as `.keras` files (optionally to Google Drive)

## Requirements
`tensorflow`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

## Limitations / next steps
- No learning-rate scheduling tested (fixed LR throughout)
- Augmentation only tested on the 3-block architecture, not the 4-block one — worth isolating that variable
- VGG16 transfer learning underperformed (61.2%) at 32×32 input resolution; worth retrying with upsized inputs or a lighter pretrained backbone (e.g. MobileNet)
