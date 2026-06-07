# signAI
https://umarjanjua.github.io/signai/

Behavioral integrity monitoring for production ML. Know your production model is still the model you validated.

---

## What signAI does

signAI monitors a model's behavioral fingerprint continuously — not its accuracy. It detects:

- **Backdoor attacks** — a model that was clean when validated but now responds to a hidden trigger
- **Gradient poisoning** — malicious training updates that subtly shift model behavior during fine-tuning
- **Supply chain tampering** — a vendor-supplied or hub-downloaded model that does not behave the way you validated it
- **Inference drift** — gradual or sudden change in activation and output distributions after redeployment

A model can maintain high accuracy while behaving maliciously on specific inputs. signAI catches this.

## Privacy model

signAI is built so privacy is enforced structurally, not by policy.

- Feature extraction runs entirely on your machine
- Only compact `s` and `z` float vectors are transmitted — typically under 100 bytes per score call
- Raw vectors are discarded after scoring
- Only `{timestamp, score, flagged, phase}` is stored in history
- **Model weights, raw inputs, and gradients never leave your machine**

You can run fully local with no server at all. The server mode adds history, alerts, and audit export — but never sees your model.

## Model support

| Status | Model type |
|--------|-----------|
| Available | PyTorch classifiers (CNN, ViT, HuggingFace classification) |
| Available | Large language models (HuggingFace generative, >200M params) |
| Roadmap | Gradient boosting (XGBoost, LightGBM, CatBoost, sklearn) |
| Roadmap | TensorFlow / Keras |
| Roadmap | JAX / Flax |
| Roadmap | ONNX (inference-only) |

New model types are added via the extractor plugin interface — detectors, server, and artifact format are framework-agnostic.

## Detectors

Three detector algorithms ship in v0.4.0:

| Kind | Algorithm | Best for |
|------|-----------|----------|
| `v1` | Conditional Mahalanobis | General purpose; fast; no GPU required |
| `nn` | Neural conditional monitor | Complex activation geometry |
| `assoc` | Blockwise association monitor | High-dimensional behavioral spaces |

## Who it is for

- **Fintech teams** verifying that credit and fraud models have not drifted since last validation
- **Healthcare teams** monitoring clinical decision support models for behavioral drift
- **MLOps teams** needing a single behavioral monitor across inference and training without exposing model data to a third-party service
- **AI security teams** auditing models for supply chain tampering or backdoor injection

## Deployment modes

| Mode | Description |
|------|-------------|
| Local artifact | Load `integrity.json` in-process. No server, no network. |
| Self-hosted server | Run inside your VPC. File or Postgres-backed. |
| Air-gapped | Private internal host. No internet required. |
| Hosted cloud | Managed endpoint. Features extracted locally. |

## Quick start

```python
from signai import monitor

# Calibrate once on clean data
m = monitor.attach(model, num_classes=10)
m.calibrate(clean_loader, device="cpu", phase="inference")
m.save("./integrity.json")

# Score every batch
m = monitor.load(model, artifact="./integrity.json")
result = m.score_inference(x, y)
if result.flagged:
    route_to_fallback(x)
```

Training monitoring:

```python
m.calibrate(train_loader, device="cpu", phase="training",
            optimizer=optimizer, criterion=criterion)
m.save("./integrity_train.json")

result = m.score_training(logits, loss)
if result.flagged:
    quarantine_update(result)
```

## Licensing

The SDK is free to use. Enterprise annual software license and support contract available for teams needing production deployment, compliance documentation, and ongoing support.

Contact: umarjanjua@live.com

## Public materials in this repo

- `index.html` — public landing page (https://umarjanjua.github.io/signai/)
- `USER_MANUAL.md` — customer-facing manual
- `DEPLOY.md` — deployment guide (local, self-hosted, air-gapped)
- `figures/` — benchmark result figures

The full SDK, server runtime, calibration and scoring implementation, tests, and release machinery live in the private `signAI-core` repository.
