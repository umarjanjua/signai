# signAI

Runtime behavioral integrity monitoring for PyTorch models.

signAI watches *how* a model behaves rather than how accurate it is, so you can detect manipulation while it happens — without exposing model weights, raw inputs, or gradients.

> This repository is the public product site and customer documentation. The implementation is private. Everything here describes how to use signAI, not how it is built.

## What signAI detects

- adversarial inputs (PGD, FGSM) and out-of-distribution traffic at inference
- gradient manipulation and targeted poisoning during training
- backdoor triggers and supply-chain weight substitution
- model extraction probing

Detection is based on a conditional behavioral model: for each operating state **S**, signAI learns the expected behavioral response **Z**, and flags deviations against a threshold calibrated from a χ² null distribution.

## Privacy

This is the founding constraint, enforced structurally rather than by policy:

- feature extraction runs inside your own process
- only compact `(S, Z)` float vectors ever cross a boundary — typically under 100 bytes per score
- model weights, raw inputs and gradients never leave your machine
- there is no hosted signAI service; nothing is sent to us at any point

## Install

```bash
pip install signai-sdk
```

The published package is `signai-sdk`; the import name is `signai`. Python 3.10+, PyTorch 2.1+.

For the calibration daemon:

```bash
pip install "signai-sdk[server]"
```

## How it works

signAI has two phases, and they have different requirements.

**Calibration** learns what normal looks like for your model. It streams behavioral vectors to a signAI daemon running on your machine, which fits the detector and stores the artifact. The daemon is local — `localhost:7731` — and needs no internet access.

**Scoring** runs entirely in-process against the saved artifact. No server, no network, nothing to operate. This is the part that lives in your inference and training loops.

## Quick start

### 1. Start the daemon

```bash
signai serve
```

### 2. Calibrate on clean data

```python
from signai import monitor

m = monitor.attach(model, num_classes=10, monitor_id="my-model")
m.calibrate(clean_loader, device="cuda", phase="inference", calib_batches=200)
m.save("./integrity.json")
```

`monitor_id` names the artifact so `save()` can retrieve it.

### 3. Score in production

```python
m = monitor.load(model, artifact="./integrity.json", device="cuda")

for x, y in test_loader:
    result = m.score_inference(x, y)
    if result.flagged:
        route_to_fallback(x)
```

The daemon can be stopped now. Scoring needs nothing running.

### Monitoring training

Calibrate for the training phase first — `optimizer` and `criterion` are required:

```python
m = monitor.attach(model, num_classes=10, monitor_id="my-model")
m.calibrate(
    train_loader,
    device="cuda",
    phase="training",
    optimizer=optimizer,
    criterion=criterion,
)
m.save("./integrity_train.json")
```

Then score each step, after `optimizer.step()`:

```python
result = m.score_training(logits, loss)
if result.flagged:
    quarantine_update(result)
```

## Detectors

| Kind | Algorithm | Best for |
|------|-----------|----------|
| `v1` | Conditional Mahalanobis | general purpose, fast, CPU-only, most auditable |
| `nn` | Neural conditional monitor | CNNs and ResNets; best on gradient-level training attacks |
| `assoc` | Blockwise association monitor | transformers and LLMs; best on backdoor and supply-chain |

Start with `v1`. Move to `nn` or `assoc` when threshold noise or a specific threat model justifies it.

## Model support

| Framework | Status |
|-----------|--------|
| PyTorch — CNN, ResNet, ViT | Supported |
| HuggingFace Transformers | Supported |
| Custom PyTorch architectures via extractor plugin | Supported |
| TensorFlow / Keras | Planned |
| JAX / Flax | Planned |
| ONNX | Planned |
| XGBoost / LightGBM / CatBoost | Planned |
| PyG / DGL (graph neural networks) | Planned |

LLM and generative-transformer monitoring uses a separate entry point from `monitor.attach()`. See the [user manual](USER_MANUAL.md).

## Deployment

signAI runs on a workstation, on a self-hosted server inside your VPC, or fully air-gapped. See the [deployment overview](DEPLOY.md).

## Licensing

Fresh installs get a **3-day trial** automatically — no key, no account, no card. It unlocks every detector and every feature.

```bash
signai apply-key sk_...    # activate or renew
signai status              # plan, features, limits, expiry
```

Keys are Ed25519-signed and verified offline. No phone-home at validation time, which is what makes air-gapped deployment work.

Pricing and purchase: **https://umarjanjua.github.io/signai/**

## Documentation

- [User manual](USER_MANUAL.md) — concepts, full SDK guide, LLM path, troubleshooting
- [Deployment overview](DEPLOY.md) — choosing a deployment mode

## Support

Questions, evaluations and enterprise enquiries: umarjanjua@live.com
