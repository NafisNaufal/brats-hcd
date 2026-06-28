# Research Plan: Decoder-Aware Tri-Axis Attention for 3D Brain Tumor Segmentation

## Target: ICoDSA 2026 (deadline ~June 2026)
## Hardware: L40 48GB (shared — available VRAM varies)
## Estimated total time: ~3-4 weeks training + 2-3 weeks writing

---

## Revised Research Framing

### Title (working)
**"Decoder-Aware Tri-Axis Attention for 3D Brain Tumor Segmentation: A Comparative Study on UNet++ and DeepLabV3+"**

### Problem Statement
Standard attention mechanisms (e.g., Squeeze-and-Excitation) applied inside 3D segmentation decoders have been shown to cause feature misalignment and catastrophic performance degradation in brain tumor segmentation — in one case reducing tumor core accuracy to zero (cite: Combining DeepLabV3 with Attention Mechanisms for Accurate Brain Tumor Segmentation: Insights from BraTS 2020 and a Private Clinical Dataset, 2024). Meanwhile, 3D CNNs and Vision Transformers struggle to maintain boundary precision across all three anatomical planes simultaneously.

### Research Gap (revised — much stronger now)
1. Tri-Axis Attention (TACA-RNet, 2024) was only evaluated inside a single custom architecture. Its effectiveness when integrated into widely-used decoder paradigms (UNet++, DeepLabV3+) is unknown.
2. No study has investigated whether **attention placement strategy** should differ based on decoder architecture — i.e., whether the same attention module works differently when placed at skip connections (UNet++) vs. post-ASPP (DeepLabV3+).
3. The interaction between modern encoders (Swin Transformer vs. ConvNeXt) and decoder-specific attention remains unexplored.

### Contributions (what reviewers will look for)
1. We propose **decoder-aware integration strategies** for Tri-Axis Attention:
   - For UNet++: at nested skip connections where multi-scale feature fusion occurs
   - For DeepLabV3+: post-ASPP and post-fusion, avoiding the failure mode of in-ASPP attention
2. We conduct the **first systematic evaluation** of Tri-Axis Attention across 2 encoders × 2 decoders on BraTS 2021.
3. We demonstrate that **placement strategy significantly impacts** segmentation performance, and that decoder-aware placement outperforms naive encoder-decoder bridge placement.

### Hypotheses (state these explicitly in the paper)
- H1: Tri-Axis Attention improves Dice scores for all encoder-decoder combinations compared to no attention.
- H2: The magnitude of improvement differs by decoder type (UNet++ vs. DeepLabV3+).
- H3: Tri-Axis Attention avoids the failure mode observed with SE attention in DeepLabV3+.

---

## Experiment Plan

### Phase 0: Code Changes ✅
- [x] Implement decoder-specific attention (done — attention lives inside each decoder)
- [x] Verify forward pass shapes for all 8 configurations (all pass)
- [x] Upgrade to MedNeXt-B, batch_size=2, use_checkpoint=false for L40
- [ ] Set up W&B logging, verify a single training step runs end-to-end

### Phase 1: Baseline Runs — WITHOUT Attention (Weeks 1-2)
Priority order (run these first — they're your baselines):

| Run | Encoder   | Decoder     | Attention | Est. Time |
|-----|-----------|-------------|-----------|-----------|
| B1  | Swin      | UNet++      | OFF       | ~1-2 days |
| B2  | Swin      | DeepLabV3+  | OFF       | ~1-2 days |
| B3  | MedNeXt-B | UNet++      | OFF       | ~1 day    |
| B4  | MedNeXt-B | DeepLabV3+  | OFF       | ~1 day    |

### Phase 2: Attention Runs — WITH Tri-Axis Attention (Weeks 2-3)
Run in same order for direct comparison:

| Run | Encoder   | Decoder     | Attention | Est. Time |
|-----|-----------|-------------|-----------|-----------|
| A1  | Swin      | UNet++      | ON        | ~1-2 days |
| A2  | Swin      | DeepLabV3+  | ON        | ~1-2 days |
| A3  | MedNeXt-B | UNet++      | ON        | ~1 day    |
| A4  | MedNeXt-B | DeepLabV3+  | ON        | ~1 day    |

### Phase 3: Bridge Attention Comparison (Week 3-4)
**This is required, not optional** — Contribution #3 ("decoder-aware placement outperforms naive bridge placement") cannot be substantiated without it. Pick the best-performing encoder from Phase 1–2 results for these two runs.

| Run | Encoder              | Decoder     | Attention Placement | Est. Time |
|-----|----------------------|-------------|---------------------|-----------|
| C1  | Best from Phase 1-2  | UNet++      | Bridge (old code)   | ~1-2 days |
| C2  | Best from Phase 1-2  | DeepLabV3+  | Bridge (old code)   | ~1-2 days |

This gives the 3-way table: No Attention vs. Bridge Attention vs. Decoder-Aware Attention.

**Total Phase 1-2: ~10-12 days**
**Total with Phase 3: ~14-16 days**

---

## Training Configuration (keep IDENTICAL across all runs)

```yaml
dataset: BraTS 2021
patch_size: [128, 128, 128]   # BraTS standard — keep fixed for comparability with literature
batch_size: 1                 # conservative — shared L40 means available VRAM varies
optimizer: AdamW
encoder_lr: 1.0e-4
decoder_lr: 2.0e-4
weight_decay: 1.0e-4
scheduler: CosineAnnealingLR (warmup_epochs=5, eta_min=1e-6)
max_epochs: 200               # fixed budget — NO early stopping (see note below)
loss: Dice + CrossEntropy (combined)
mixed_precision: true
use_checkpoint: true          # re-enabled as safety net for shared VRAM

# MedNeXt
model_id: B                   # S was a 12GB workaround — B is the intended model
n_channels: 32
kernel_size: 3

# Swin
feature_size: 48
pretrained: true
use_checkpoint: true          # gradient checkpointing for Swin — shared VRAM safety net

# Data
train_val_split: 5-fold cross-validation
augmentation: random flip, random rotation, random intensity shift
normalization: per-volume z-score normalization
```

### No early stopping — train all runs to exactly 200 epochs
Early stopping would let different runs terminate at different epochs, introducing a confound into the comparison. With a fixed training budget and L40 throughput, just run all 10 runs to completion. This also makes the results table cleaner — every row was trained identically.

### 5-fold cross-validation
On the 3060, a single 80/20 split was the only feasible option. On the L40, 5-fold CV is practical within the timeline and is required for the paper to survive reviewer scrutiny on a comparative study. Report mean ± std across folds.

### Critical: Keep these FIXED across all experiments
- Same 5-fold split indices
- Same augmentation pipeline
- Same learning rate, optimizer, scheduler
- Same patch size and batch size
- Same loss function
- Same random seed

The ONLY variable should be: encoder type, decoder type, and attention placement.

---

## Evaluation Metrics

### Primary metrics (report for each tumor region):
- **Dice Similarity Coefficient (DSC)** for WT (Whole Tumor), TC (Tumor Core), ET (Enhancing Tumor)
- **Hausdorff Distance 95% (HD95)** for WT, TC, ET

### How to report:
- 5-fold CV: report mean ± std across folds for all 10 runs
- **Statistical test**: Wilcoxon signed-rank test (paired, non-parametric) between with/without attention for each combo. Report p-values. p < 0.05 = significant.

### Baselines to cite (don't need to re-run, just reference published numbers):
- Vanilla 3D U-Net on BraTS 2021 (from BraTS challenge leaderboard)
- Swin UNETR (from the original paper — they report BraTS 2021 results)
- nnU-Net (from the BraTS 2021 challenge)
- TACA-RNet (from the original paper — BraTS 2018/2020 results)

---

## Paper Structure (ICoDSA format)

### 1. Introduction (~1 page)
- Brain tumor segmentation importance
- Problem: attention mechanisms fail in 3D segmentation (cite IEEE paper)
- Gap: no decoder-aware attention integration study
- Contributions (3 bullet points from above)

### 2. Related Work (~1 page)
- Brain tumor segmentation (BraTS challenge overview)
- UNet++ and DeepLabV3+ architectures
- Attention mechanisms in medical imaging (SE, CBAM, Tri-Axis)
- Position this work: "Unlike prior work that applies attention uniformly, we investigate decoder-specific placement."

### 3. Methodology (~2 pages)
- 3.1 Overall architecture diagram (encoder → decoder-specific attention → output)
- 3.2 Tri-Axis Attention module (TACR + AxisAttention3D) — describe, cite TACA-RNet
- 3.3 Integration into UNet++ (where and why — at nested skip connections)
- 3.4 Integration into DeepLabV3+ (where and why — post-ASPP and post-fusion)
- 3.5 Encoder variants (Swin Transformer, ConvNeXt/MedNeXt)
- **Architecture diagrams are required, not nice-to-have.** The UNet++ j-indexed dense node grid and DeepLabV3+ ASPP placement cannot be described clearly in prose alone within ICoDSA page limits. Plan dedicated time for these figures.

### 4. Experimental Setup (~1 page)
- Dataset: BraTS 2021
- Preprocessing and augmentation
- Training details (AdamW, LR, 200 epochs fixed, 5-fold CV, L40 48GB)
- Evaluation metrics

### 5. Results and Discussion (~2 pages)
- Table 1: All 8 configurations — Dice (WT, TC, ET) and HD95, mean ± std
- Table 2: Statistical significance (Wilcoxon p-values for attention vs. no attention)
- Table 3: Bridge vs. Decoder-Aware attention comparison (Phase 3)
- Discussion: Which decoder benefits more from attention? Why?
- Discussion: Swin vs. MedNeXt — which encoder pairs better?
- Qualitative results: Show 2-3 segmentation visualizations (good case, hard case, failure case)

### 6. Conclusion (~0.5 page)
- Summary of findings
- Limitation: single dataset
- Future work: test on BraTS 2023/2024, other attention modules

---

## Timeline

| Week | Activity |
|------|----------|
| Week 1   | Verify training pipeline end-to-end, run Phase 1 baselines (B1-B4) |
| Week 2   | Run Phase 2 attention runs (A1-A4) |
| Week 3   | Run Phase 3 bridge comparison (C1-C2), analyze results |
| Week 4   | Generate tables, figures, architecture diagrams |
| Week 5-6 | Write paper draft |
| Week 7-8 | Revise, get feedback, polish, submit |

---

## Checklist Before Submission

- [ ] All 10 runs completed (B1-B4, A1-A4, C1-C2) with consistent settings
- [ ] 5-fold CV completed — results reported as mean ± std
- [ ] Statistical tests computed (Wilcoxon signed-rank)
- [ ] Architecture diagrams created (overall + UNet++ dense node grid + DeepLab ASPP placement)
- [ ] Segmentation visualizations generated (overlay on MRI slices)
- [ ] Results table includes mean ± std
- [ ] Hardware and training details fully documented
- [ ] All references from your paper tracker properly cited
- [ ] Code cleaned up and ready to share (GitHub link in paper)
