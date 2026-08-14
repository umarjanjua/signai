# signAI User Manual

*Customer documentation. The implementation lives in a private repository; this manual covers using signAI, not building it.*

## Overview

signAI is a runtime behavioral integrity monitor for PyTorch models. It detects suspicious model behavior during:

- **inference** — input anomaly, distribution shift, activation drift
- **training** — gradient manipulation, targeted poisoning

It runs on a developer workstation, on a self-hosted server inside your network, or fully air-gapped. There is no hosted signAI service — nothing leaves your infrastructure at any point.

---

## Core Concepts

### Conditional Behavioral Model (CBM)

For each operating state S, signAI learns the expected behavioral response Z and scores deviations. High deviation means potential integrity loss.

- **S (conditioning vector)** — what the model sees: input statistics, prediction confidence, loss level
- **Z (behavioral vector)** — how the model responds: activation patterns, gradient geometry, layer norms
- **Detector** — fits P(Z|S) on clean calibration data, then scores new observations at runtime

### Artifact

A JSON file produced by calibration. It holds detector parameters and thresholds — **not** model weights. Typically 50 KB to 2 MB depending on detector kind.

### The daemon

A local signAI service on `localhost:7731`. Calibration streams `(S, Z)` vectors to it; it fits the detector and stores the artifact. Start it with `signai serve`. It needs no internet access.

### Local artifact scoring

Once an artifact exists, the SDK loads it and scores in-process — no server, no network. This path cannot calibrate; it scores against a detector fitted earlier.

### Detector kinds

- `v1` — conditional Mahalanobis; fast, CPU-only, most auditable threshold argument
- `nn` — neural conditional monitor; better on complex activation geometry
- `assoc` — blockwise association monitor; for transformers and high-dimensional behavioral spaces

---

## Setup Paths

### Path 1: Workstation — calibrate locally, then score offline

Best for notebooks, experimentation, and teams that don't need centralized history.

1. Start the daemon: `signai serve`
2. Attach a monitor with a `monitor_id`
3. Calibrate on clean data
4. Save the artifact
5. Stop the daemon — load the artifact and score in-process from here on

### Path 2: Self-hosted server

Best for platform teams, central alerting, VPC deployment, fleet-wide monitor management.

1. Run signAI server on a shared host inside your network
2. Calibrate against a daemon and save the artifact
3. Upload the artifact to the server
4. Point monitors at the server endpoint and score

### Path 3: Air-gapped

Path 2, with the server on a private host that has no internet access. License keys validate offline, so no connectivity is required at any point.

---

## SDK Guide

### Create a monitor

```python
from signai import monitor

m = monitor.attach(
    model,
    num_classes=10,
    monitor_id="my-model",
    device="cuda",
)
```

`monitor_id` names the artifact on the daemon. `save()` retrieves the artifact by that name, so a monitor created without one cannot save.

`monitor.attach()` covers classification models. For LLMs and generative transformers, see *Working with LLMs* below.

### Load an existing monitor

```python
m = monitor.load(model, artifact="./integrity.json", device="cuda")
```

Against a self-hosted server:

```python
m = monitor.load(
    model,
    endpoint="https://signai.internal:8000",
    monitor_id="my-model",
    api_key="...",
    device="cuda",
)
```

---

## Calibration

Calibrate on clean, representative data. Requires a running daemon.

### Inference

```python
m.calibrate(
    clean_loader,
    device="cuda",
    phase="inference",
    calib_batches=200,
)
```

### Training

`optimizer` and `criterion` are required:

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

### Save

```python
m.save("./integrity.json")
```

This downloads the calibrated artifact from the daemon, writes it to disk, and switches the monitor to local scoring.

### How much data

| Detector | Minimum samples | Recommended |
|----------|----------------|-------------|
| `v1` | 100 | 500+ |
| `nn` | 200 | 1,000+ |
| `assoc` | 300 | 1,000+ |

Calibration is a one-time cost per model — re-run it only after retraining.

---

## Runtime Scoring

### Inference

```python
result = m.score_inference(x, y)
if result.flagged:
    route_to_fallback(x)
```

### Training

Call after `optimizer.step()` so parameter deltas are captured:

```python
result = m.score_training(logits, loss)
if result.flagged:
    quarantine_update(result)
```

Pass `logits` and `loss` — not `x`, `y`, or the optimizer.

### MonitorResult

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

Scoring is designed not to raise in hot paths. If a remote call fails, you get a result with `error` populated rather than an exception.

### Overhead

| Detector | Per-inference cost |
|----------|-------------------|
| `v1` | < 0.05 ms |
| `nn` | < 0.3 ms |
| `assoc` | < 0.5 ms |

Negligible against any model forward pass.

---

## Working with LLMs

Monitoring for LLMs and generative transformers uses a different entry point from `monitor.attach()`, which covers classification models. The LLM path uses per-module L2 norm deltas rather than full parameter flattening, so memory stays constant regardless of model size — practical for models in the 7B–70B range.

Contact umarjanjua@live.com for the LLM onboarding guide and a worked example for your architecture.

---

## Licensing

A valid license gates all usage — calibration, scoring, history and alerts.

Fresh installs get a **14-day trial** automatically. No key, no account, no card. The trial unlocks every detector and every feature, with a 1-model limit and 7 days of scoring history. Paid plans start at 3 models and 30 days of history.

### Apply a key

```bash
signai apply-key sk_...
```

Saved to `~/.signai/license.json` (or `$SIGNAI_HOME/license.json`).

### Check status

```bash
signai status
```

Shows status, seat, plan, features, model limit, history retention and expiry. Calibrations are not metered — there is no per-month usage count.

### Renew

Purchase or renew at https://umarjanjua.github.io/signai/, then run `signai apply-key` with the new key. It replaces the previous one immediately.

Keys are Ed25519-signed and verified offline — no network call at validation time.

---

## Troubleshooting

### `No signAI daemon found. Start it with: signai serve`

`calibrate()` or `save()` was called with no daemon reachable on `localhost:7731`. Start it and retry.

### `save()` returns a 404 for `/v1/artifacts//download`

The monitor was created without a `monitor_id`, so there is no artifact name to fetch:

```python
m = monitor.attach(model, num_classes=10, monitor_id="my-model")
```

### `Monitor is not calibrated. Call calibrate() before scoring.`

You attached a monitor and scored it without calibrating, or calibration did not complete. Calibrate, then save.

### `optimizer and criterion are required for training calibration`

Training calibration needs both:

```python
m.calibrate(loader, device="cuda", phase="training",
            optimizer=optimizer, criterion=criterion)
```

### `Torch not compiled with CUDA enabled`

Pass `device="cpu"` to `calibrate()`. All three detectors score on CPU; a GPU only speeds up `nn` and `assoc` calibration.

### 402 response on history or scoring

No valid license is active — the trial has expired and no key is applied. Run `signai apply-key sk_...`.

### Scores are always near zero

Calibration data was too small or unrepresentative. Increase `calib_batches` or use more diverse clean samples.

---

## Operational Notes

- scoring endpoints are designed not to raise in hot paths; setup operations such as artifact upload can raise
- calibrate on representative clean traffic — threshold quality follows directly from this
- servers store `{ts, score, flagged, phase}` only, never raw behavioral vectors
- re-calibrate after retraining; a stale artifact describes a model that no longer exists
