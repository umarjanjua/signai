# signAI

A minimal public artifact for **input-conditioned runtime integrity monitoring** 

## What this public repo shows

This public slice is intentionally narrow:
- one publishable summary figure
- a few selected results images

It is designed to communicate the core idea without exposing the full `signai-core` system.
Please request access at umarjanjua@live.com

## Core idea

signAI models the relationship between an input-derived signature `S` and an internal behavior signature `Z`.
At inference time, the strongest public story is the **I1 monitor**:
- `S`: simple input statistics
- `Z`: depth-stratified activation statistics
- anomaly score: conditional Mahalanobis residual

The interpretability bridge is:

> if internal behavior is structured enough to model from inputs, then interpretability is not only post-hoc explanation; it can also become a runtime signal for integrity monitoring.


- strong quantitative signal
- simple visual comparison against established baselines
- no need to expose the full training pipeline

## Included assets

- `figures/public_results_snapshot.png` — summary figure for sharing
- `figures/roc_i1_svhn.png` — I1 ROC on SVHN near-OOD
- `figures/roc_mahal_svhn.png` — Mahalanobis ROC on SVHN near-OOD
- `figures/roc_energy_svhn.png` — Energy ROC on SVHN near-OOD


> signAI shows that internal model behavior can be modeled conditionally from inputs strongly enough to support runtime integrity monitoring, with especially strong results for inference-time detection.

## Full system

The complete system, extended experiments are maintained separately.
Access is available on request.
