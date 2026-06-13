# signAI User Manual

## Overview

signAI is a runtime behavioral integrity monitor for PyTorch models. It helps teams detect suspicious model behavior during:

- inference (input anomaly, distribution shift, activation drift)
- training (gradient manipulation, targeted poisoning)
- managed cloud deployments
- self-hosted and air-gapped environments

The production layer has one SDK and one server contract. The deployment mode changes only how scoring is routed.

## Core Concepts

### Conditional Behavioral Model (CBM)

For each operating state S, signAI learns the expected behavioral response Z and scores deviations. High deviation scores indicate potential integrity issues.

- **S (conditioning vector)**: captures what the model sees — input statistics, prediction confidence, loss level
- **Z (behavioral vector)**: captures how the model responds — activation patterns, gradient geometry, layer norms
- **Detector**: fits P(Z|S) on clean calibration data; scores new observations at runtime

### Extractor

An extractor produces (S, Z) pairs from model internals. signAI ships two extractors and supports custom ones:

- `ClassificationExtractor`: for CNNs, ViTs, and HuggingFace classifiers
- `LLMExtractor`: for LLMs and generative transformers

### Detector kinds

The detector implements the anomaly scoring function. Three kinds are available:

- `v1`: conditional Mahalanobis distance (sklearn-based, fast, no GPU required)
- `nn`: neural conditional monitor (higher accuracy on complex activation geometry)
- `assoc`: blockwise association neural monitor (high-dimensional behavioral spaces)

### Monitor

The SDK object you attach to a model:

```python
from signai import monitor
```

### Artifact

A JSON file generated after calibration. It stores detector parameters and thresholds, not model weights.

### Local mode

The SDK loads the artifact and scores in-process. No server is required.

### Remote mode

The SDK extracts behavioral vectors locally and sends only those vectors to a server endpoint.

---

## Customer Setup Paths

### Path 1: Local-only

Best for:

- notebooks
- experimentation
- offline use
- teams that do not need centralized history

Workflow:

1. Attach a monitor
2. Calibrate on clean data
3. Save the artifact
4. Load the artifact later and score locally

### Path 2: Self-hosted server

Best for:

- internal platform teams
- central alerting
- VPC deployment
- fleet-wide monitor management

Workflow:

1. Start `signai-server`
2. Attach a monitor with `endpoint=...`
3. Calibrate locally
4. Save the artifact, which also uploads it
5. Score using the same `score_inference` and `score_training` calls

### Path 3: Air-gapped

Same as self-hosted, but the endpoint lives on a private host with no internet access.

---

## SDK Guide

### Create a monitor

```python
from signai import monitor

m = monitor.attach(
    model,
    num_classes=10,
    device="cuda",
)
```

The extractor is auto-selected: models above 200M parameters or HuggingFace generative models receive `LLMExtractor`; all others receive `ClassificationExtractor`.

### Load an existing monitor

Local:

```python
m = monitor.load(model, artifact="./integrity.json", device="cuda")
```

Remote:

```python
m = monitor.load(
    model,
    endpoint="http://localhost:8000",
    monitor_id="my-model",
    api_key="",
    device="cuda",
)
```

---

## Calibration

Calibration should use clean, representative data.

### Inference calibration

```python
m.calibrate(
    clean_loader,
    device="cuda",
    phase="inference",
    calib_batches=200,
)
```

### Training calibration

```python
m.calibrate(
    train_loader,
    device="cuda",
    phase="training",
    optimizer=optimizer,
    criterion=criterion,
    warmup_steps=200,
)
```

### Save the artifact

```python
m.save("./integrity.json")
```

In remote mode, `save()` also uploads the artifact to the configured server.

---

## Runtime Scoring

### Inference

```python
result = m.score_inference(x, y)
if result.flagged:
    route_to_fallback(x)
```

### Training

Call after `optimizer.step()`:

```python
result = m.score_training(logits, loss)
if result.flagged:
    quarantine_update(result)
```

Important:

- pass `logits` and `loss`
- do not pass `x`, `y`, or `optimizer` into `score_training`
- call after `optimizer.step()` so parameter deltas are captured

---

## MonitorResult

The SDK returns:

```python
MonitorResult(
    score=float,
    flagged=bool,
    tau=float,
    phase="inference" | "training",
    monitor_id=str,
    error=str | None,
)
```

If a remote scoring call fails, the SDK returns a safe result with `error` populated instead of raising.

---

## Working with LLMs and Large Models

For models above 200M parameters or HuggingFace generative models, use `IntegrityMonitor` and `LLMExtractor` directly.

### Calibrate a language model

```python
from signai import IntegrityMonitor, LLMExtractor

extractor = LLMExtractor(llm)
m = IntegrityMonitor(llm, num_classes=None, extractor=extractor, detector_kind="v1")
m.calibrate_inference(clean_loader, calib_batches=200, device="cuda")
m.export_json("./integrity_llm.json")
```

### Score LLM inference

```python
m = IntegrityMonitor(llm, num_classes=None, extractor=LLMExtractor(llm), detector_kind="v1")
m.import_json("./integrity_llm.json")

for batch in eval_loader:
    result = m.score_input(batch["input_ids"], None, device="cuda")
    if result.flagged:
        escalate(batch)
```

### LLMExtractor configuration

```python
extractor = LLMExtractor(
    model,
    max_tracked_layers=24,   # how many module layers to track; default 24
    use_sensitivity=False,   # sensitivity backward pass; disabled by default for LLMs
)
```

The LLMExtractor tracks per-module L2 norm deltas rather than full parameter flattening. This keeps memory constant at O(max_tracked_layers) scalars per step, regardless of model size.

### HuggingFace models

Auto-selection works with HuggingFace models. If `model.config.model_type` is present and `num_classes` is not provided, `LLMExtractor` is selected automatically:

```python
from transformers import AutoModelForCausalLM
from signai import SignatureExtractorBase

model = AutoModelForCausalLM.from_pretrained("gpt2")
extractor = SignatureExtractorBase.auto(model)  # returns LLMExtractor
```

---

## Detector Kinds

The detector kind determines the anomaly scoring algorithm. Set it via `IntegrityMonitor` or the server calibration API.

### v1 — Conditional Mahalanobis

Default. Fast, no GPU required, no neural training needed.

```python
from signai import IntegrityMonitor, ClassificationExtractor

m = IntegrityMonitor(model, num_classes=10,
                     extractor=ClassificationExtractor(model),
                     detector_kind="v1")
```

### nn — Neural Conditional Monitor

Neural network-based. Better at capturing complex dependencies in the Z space.

```python
m = IntegrityMonitor(model, num_classes=10,
                     extractor=ClassificationExtractor(model),
                     detector_kind="nn",
                     detector_config={"hidden_dim": 64, "epochs": 30})
```

### assoc — Blockwise Association Neural Monitor

For high-dimensional behavioral spaces where Z dimension is large.

```python
m = IntegrityMonitor(model, num_classes=10,
                     extractor=ClassificationExtractor(model),
                     detector_kind="assoc",
                     detector_config={"block_size": 8, "epochs": 30})
```

### Detector kind and artifact format

Each kind writes a distinct artifact format:

| Kind | Artifact format string |
|------|----------------------|
| `v1` | `signai_integrity_v1` |
| `nn` | `signai_integrity_nn_v1` |
| `assoc` | `signai_integrity_assoc_v1` |

The artifact format is stored in the JSON file and loaded transparently by `monitor.load()`.

---

## Direct IntegrityMonitor API (Advanced)

Use `IntegrityMonitor` directly when you need full control over extractor, detector, and calibration flow.

### Collect raw (S, Z) arrays

```python
from signai import IntegrityMonitor, ClassificationExtractor

m = IntegrityMonitor(model, num_classes=10,
                     extractor=ClassificationExtractor(model),
                     detector_kind="v1")

# Collect without fitting
S, Z = m.collect_inference_batches(clean_loader, calib_batches=200, device="cuda")

# Fit manually
m.det_infer.fit(S, Z)
m.export_json("./integrity.json")
```

### Score directly

`score_input` and `score_update` take raw model inputs and extract internally:

```python
# inference — x and y are raw tensors, device is where the model runs
result = m.score_input(x, y, device="cuda")
```

```python
# training — call after optimizer.step()
result = m.score_update(logits, loss)
```

To access the raw `(s, z)` vectors (e.g. for logging or custom thresholds):

```python
s, z = m.extractor.extract_inference(x, y, device="cuda")
det_score = m.det_infer.score(s, z)
```

---

## Custom Extractor Plugins

To support a model architecture that neither `ClassificationExtractor` nor `LLMExtractor` covers:

```python
from signai import SignatureExtractorBase, IntegrityMonitor
import numpy as np

class TabularExtractor(SignatureExtractorBase):
    def __init__(self, model):
        self._model = model
        self._ema = None

    def prepare(self, example_batch):
        pass  # tabular models need no hook setup

    def reset_state(self):
        self._ema = None

    def extract_training(self, logits, loss):
        # s: [loss, entropy]
        # z: [delta_l2, ema_cos]
        ...
        return s.astype(np.float32), z.astype(np.float32)

    def extract_inference(self, x, y, device="cpu", use_sensitivity=True):
        ...
        return s.astype(np.float32), z.astype(np.float32)

# Use it
m = IntegrityMonitor(model, num_classes=2,
                     extractor=TabularExtractor(model),
                     detector_kind="v1")
```

The `extract_training` and `extract_inference` methods must return `float32` numpy arrays with consistent dimensions across calls.

---

## Server Operation

### Start a file-backed server

```bash
signai-server serve --host 0.0.0.0 --port 8000 --storage ./artifacts
```

### Start with SQLite

```bash
signai-server serve \
  --host 0.0.0.0 \
  --port 8000 \
  --store-backend sqlite \
  --database-url sqlite:///./signai.db
```

### Start with Postgres

```bash
signai-server serve \
  --host 0.0.0.0 \
  --port 8000 \
  --store-backend postgres \
  --database-url postgresql://user:pass@host:5432/signai
```

---

## Authentication

Set an API key:

```bash
signai-server serve --api-key "your-secret"
```

Then connect with:

```python
m = monitor.load(
    model,
    endpoint="http://localhost:8000",
    api_key="your-secret",
    monitor_id="my-model",
)
```

---

## Docker

Build:

```bash
docker build -t signai .
```

Run:

```bash
docker run -p 8000:8000 -v %cd%\artifacts:/data signai
```

Compose:

```bash
docker compose up --build
```

---

## Recommended Customer Rollout

1. Start in local artifact mode with the default v1 detector
2. Validate thresholds on clean and known-bad traffic
3. Move to self-hosted server mode if history or centralized operations are needed
4. Add notifier configuration for flagged events
5. Evaluate nn or assoc detectors if v1 thresholds are noisy on your model
6. Move to SQLite or Postgres storage when audit history retention matters

---

## Licensing

A license key gates all usage (calibration, scoring, history, alerts).

A bundled trial key is active on fresh installs — no action needed to get started.

### Apply a key

```bash
signai apply-key sk_...
```

The key is saved to `~/.signai/license.json` (or `$SIGNAI_HOME/license.json`).

### Check status

```bash
signai status
```

Shows seat ID, plan, features, expiry date, and calibrations used this month.

### Renew or upgrade

Visit https://umarjanjua.github.io/signai/ to purchase a new key. Run `signai apply-key sk_...` with the new key — it replaces the previous one immediately.

Keys are scoped by features and expiry date at issuance time.

---

## Troubleshooting

### `No backend configured`

Cause: you attached and calibrated but did not save.

```python
m.save("./integrity.json")
```

### Remote scoring returns `error`

Cause: endpoint unavailable, bad API key, or artifact not uploaded yet.

- verify server health at `/health`
- verify `monitor_id`
- verify `m.save(...)` was called in remote mode

### 402 response on history/status endpoints

Cause: no valid license key is active.

Apply a license key and restart:

```bash
signai apply-key sk_...
signai-server serve ...
```

### Scores are always near zero

Cause: calibration data was too small or not representative.

Increase `calib_batches` or use more diverse clean samples.

### LLMExtractor gives all-zero Z vectors

Cause: the model has no named children (e.g., a single wrapped module).

Check `list(model.named_children())`. If empty, the extractor tracks root-level parameters. This is expected and produces a valid but simplified Z.

---

## Operational Notes

- scoring endpoints are designed not to raise in hot paths
- setup operations such as artifact upload can raise on failure
- remote servers store scores and flags, not raw behavioral vectors
- calibrate on representative clean traffic for best thresholds
- sensitivity backward pass is disabled in `LLMExtractor` by default (too expensive for large models)
