# signAI

Runtime behavioral integrity monitoring for PyTorch models.

signAI adds a small monitoring layer around training and inference so you can detect suspicious model behavior in real time — without exposing model weights, raw inputs, or gradients. The production layer supports:

- Local artifact mode with no server
- Daemon mode (localhost:7731, auto-discovered by the SDK)
- Self-hosted server mode
- Air-gapped private deployment mode

## What signAI detects

signAI monitors a model's behavioral fingerprint rather than its accuracy. It catches anomalies such as:

- distribution shift in inputs or activations
- gradient manipulation and targeted poisoning during training
- unexpected output distribution changes at inference time

Detection is based on a conditional behavioral model (CBM): for each operating state S, signAI learns the expected behavioral response Z and flags deviations.

## Install

SDK only, local mode:

```bash
pip install signai
```

SDK plus server dependencies:

```bash
pip install "signai[server]"
```

SDK plus server plus Postgres support:

```bash
pip install "signai[server,postgres]"
```

## Quick Start

### Local mode (any classifier)

```python
from signai import monitor

m = monitor.attach(model, num_classes=10)
m.calibrate(clean_loader, device="cuda", phase="inference", calib_batches=200)
m.save("./integrity.json")
```

Load and score:

```python
m = monitor.load(model, artifact="./integrity.json", device="cuda")

for x, y in test_loader:
    result = m.score_inference(x, y)
    if result.flagged:
        route_to_fallback(x)
```

### LLM / large model quick start

For models above 200M parameters or HuggingFace generative models, use `LLMExtractor` and `IntegrityMonitor` directly:

```python
from signai import IntegrityMonitor, LLMExtractor

extractor = LLMExtractor(llm)
m = IntegrityMonitor(llm, num_classes=None, extractor=extractor, detector_kind="v1")
m.calibrate_inference(clean_loader, calib_batches=200, device="cuda")
m.export_json("./integrity_llm.json")
```

Score LLM inference:

```python
m = IntegrityMonitor(llm, num_classes=None, extractor=LLMExtractor(llm), detector_kind="v1")
m.import_json("./integrity_llm.json")

for batch in eval_loader:
    result = m.score_input(batch["input_ids"], None, device="cuda")
```

### Remote mode

Start the server:

```bash
signai-server serve --host 0.0.0.0 --port 8000 --storage ./artifacts
```

Connect the SDK:

```python
from signai import monitor

m = monitor.attach(
    model,
    num_classes=10,
    endpoint="http://localhost:8000",
    api_key="",
    monitor_id="demo-model",
    device="cuda",
)
m.calibrate(clean_loader, device="cuda", phase="inference", calib_batches=200)
m.save("./integrity.json")
```

## Detector Kinds

signAI ships three detector algorithms. Choose by setting `detector_kind`:

| Kind | Algorithm | Best for |
|------|-----------|----------|
| `"v1"` | Conditional Mahalanobis (sklearn) | general purpose; fast; no GPU required |
| `"nn"` | Neural conditional monitor | complex activation geometry; higher accuracy |
| `"assoc"` | Blockwise association neural monitor | high-dimensional behavioral spaces |

Use `v1` as the default. Use `nn` or `assoc` when you need finer discrimination on complex models.

```python
from signai import IntegrityMonitor, ClassificationExtractor

# Neural detector
m = IntegrityMonitor(
    model,
    num_classes=10,
    extractor=ClassificationExtractor(model),
    detector_kind="nn",
)
```

## Extractor Kinds

signAI uses an extractor plugin system to support different model types. The right extractor is auto-selected when you use `monitor.attach()`:

| Extractor | Auto-selected when | Description |
|-----------|-------------------|-------------|
| `ClassificationExtractor` | ≤200M params, classification | CNN/ViT classifier; tracks gradient geometry and activation depth |
| `LLMExtractor` | >200M params or HF generative | Per-module L2 norm delta tracking; constant memory |

To override auto-selection:

```python
from signai import IntegrityMonitor, LLMExtractor

# Force LLMExtractor on a model below the size threshold
m = IntegrityMonitor(model, extractor=LLMExtractor(model), detector_kind="v1")
```

To implement a custom extractor for a novel architecture:

```python
from signai import SignatureExtractorBase
import numpy as np

class MyExtractor(SignatureExtractorBase):
    def prepare(self, example_batch): ...
    def reset_state(self): ...
    def extract_training(self, logits, loss) -> tuple[np.ndarray, np.ndarray]: ...
    def extract_inference(self, x, y, device="cpu", use_sensitivity=True) -> tuple[np.ndarray, np.ndarray]: ...
```

## Deployment Modes

### 1. Local artifact

```python
m = monitor.load(model, artifact="./integrity.json")
```

### 2. Daemon (localhost:7731)

The SDK auto-discovers a running `signai-server` on `localhost:7731`. No endpoint config needed:

```python
m = monitor.attach(model, num_classes=10)  # connects to daemon if running
```

### 3. Self-hosted

```python
m = monitor.load(
    model,
    endpoint="http://signai.internal:8000",
    api_key="...",
    monitor_id="my-model",
)
```

### 4. Air-gapped

```python
m = monitor.load(
    model,
    endpoint="http://10.0.1.5:8000",
    api_key="...",
    monitor_id="my-model",
)
```

## Training Loop Integration

First calibrate for the training phase — `optimizer` and `criterion` are required:

```python
from signai import monitor
import torch

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

m = monitor.attach(model, num_classes=10)
m.calibrate(
    train_loader,
    device="cpu",
    phase="training",
    optimizer=optimizer,
    criterion=criterion,
)
m.save("./integrity_train.json")
```

Then score each training step after `optimizer.step()`:

```python
m = monitor.load(model, artifact="./integrity_train.json", device="cpu")

for x, y in train_loader:
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    result = m.score_training(logits, loss)
    if result.flagged:
        quarantine_update(result)
```

## Inference Loop Integration

```python
from signai import monitor

m = monitor.load(model, artifact="./integrity.json", device="cuda")

for x, y in test_loader:
    result = m.score_inference(x, y)
    if result.flagged:
        route_to_fallback(x)
```

## Public API

Top-level exports from `signai`:

| Symbol | Description |
|--------|-------------|
| `monitor.attach(model, num_classes, ...)` | Create a new monitor |
| `monitor.load(model, artifact, ...)` | Load from artifact or remote |
| `Monitor` | Monitor class with `calibrate`, `save`, `score_inference`, `score_training` |
| `MonitorResult` | Score result dataclass |
| `IntegrityMonitor` | Advanced: unified monitor with extractor + detector_kind params |
| `SignatureExtractorBase` | Advanced: ABC for custom extractor plugins |
| `ClassificationExtractor` | Classifier extractor (CNN, ViT, HF classification) |
| `LLMExtractor` | LLM extractor (generative transformers, >200M param models) |

## Run The Server

### CLI

```bash
signai-server serve \
  --host 0.0.0.0 \
  --port 8000 \
  --storage ./artifacts \
  --store-backend file
```

Shortcut through the main CLI:

```bash
signai serve --host 0.0.0.0 --port 8000 --storage ./artifacts
```

### Docker

```bash
docker build -t signai .
docker run -p 8000:8000 -v %cd%\artifacts:/data signai
```

### docker-compose

```bash
docker compose up --build
```

## Licensing

All usage requires a license key. A bundled trial key is active on fresh installs — no purchase needed to get started.

```bash
signai apply-key sk_...    # activate or renew a key
signai status              # show seat, plan, features, expiry, usage
```

Visit https://umarjanjua.github.io/signai/ to purchase or renew.

## API Summary

Server endpoints:

- `GET /health`
- `POST /v1/artifacts`
- `GET /v1/artifacts/{monitor_id}`
- `DELETE /v1/artifacts/{monitor_id}`
- `POST /v1/score/inference`
- `POST /v1/score/training`
- `POST /v1/score/batch`
- `POST /v1/calibrate/start`
- `POST /v1/calibrate/push`
- `POST /v1/calibrate/commit`
- `GET /v1/history/{monitor_id}`
- `POST /v1/notify/configure`
- `GET /v1/audit/export`
- `GET /v1/status`
- `GET /v1/monitors`
- `POST /v1/license`

## Model Support

| Framework | Status |
|-----------|--------|
| PyTorch (CNN, ViT, ResNet, etc.) | ✅ Supported |
| HuggingFace Transformers (BERT, GPT, LLaMA, etc.) | ✅ Supported |
| Custom PyTorch architectures via extractor plugin | ✅ Supported |
| TensorFlow / Keras | Planned |
| JAX / Flax | Planned |
| ONNX | Planned |
| XGBoost / LightGBM / CatBoost | Planned |
| PyG / DGL (graph neural networks) | Planned |

## Privacy Model

signAI is built so that privacy is enforced structurally:

- feature extraction runs on the customer machine
- remote scoring receives only `s` and `z` float vectors (compact; typically <100 bytes per score call)
- the server discards raw vectors after scoring
- only `{ts, score, flagged, phase}` is stored in history
- model weights, raw inputs, and gradients never leave the customer machine

## Documentation

- User manual: [USER_MANUAL.md](USER_MANUAL.md)
- Deployment guide: [DEPLOY.md](DEPLOY.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)

## License

Commercial license required for production use. A 3-day trial is included on fresh installs — no purchase needed to evaluate.

Run `signai status` to check your current license state.

## Early Access

signAI is currently in selective early access. Applications are reviewed individually.

**[Apply for access and see pricing →](https://umarjanjua.github.io/signai/#pricing)**

Or email `umarjanjua@live.com` with your use case, team size, and preferred deployment mode.
