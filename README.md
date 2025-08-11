# DualLoc — Full-Parameter Fine-Tuning of Dual Transformer Models for Protein Subcellular Localization

**DualLoc** is a dual-encoder Transformer framework for multi-label protein subcellular localization and sorting-signal prediction. By performing full-parameter fine-tuning on paired protein language models (ProtBERT, ESM-2, ProtT5) and integrating attention + dropout modules, DualLoc improves localization accuracy across ten compartments and enhances signal-peptide interpretability compared with lightweight methods (e.g., DeepLoc 2.0).

**Original article:** *DualLoc: Full-Parameter Fine-Tuning of Dual Transformer Models for Protein Subcellular Localization Prediction* — Yan Guang Chen et al.

---

## Highlights
- **Dual-encoder design:** one pretrained PLM + one randomly initialized PLM per pathway.  
- **End-to-end full-parameter fine-tuning** (ProtBERT / ESM-2 / ProtT5).  
- **Hierarchical prediction:** 10 subcellular localization labels (primary) → 9 sorting-signal classes (auxiliary).  
- **Strong empirical gains** on Swiss-Prot cross-validation and independent HPA validation.  
  - Example: **DualLoc-ProtT5 (Swiss-Prot)** — Accuracy **0.5872**, Micro-F1 **0.8371**, Macro-F1 **0.7811**.  
- PMI and attention analyses reveal biologically meaningful co-occurrence and signal localization patterns.

---

## Features

- Multilabel localization across 10 compartments: cytoplasm, nucleus, extracellular, cell membrane, mitochondrion, plastid, endoplasmic reticulum (ER), lysosome/vacuole, Golgi, peroxisome.

- Sorting-signal classification for 9 signal types: SP, TM, MT, CH, TH, NLS, NES, PTS, GPI.

- Attention-based interpretability (uses KL divergence to assess alignment between attention and true signal positions).

- Supports embedding visualization (PCA / UMAP / t-SNE).

- Supports multiple PLM backbones (ProtBERT, ESM-2, ProtT5).
---
