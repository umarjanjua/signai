# signAI User Manual

## Purpose

This manual is the public customer-facing guide for signAI.

The `signai` repository is the public front for the private `signAI-core` product codebase. It explains how the product works, how customers use it, and how deployments are typically structured without exposing private implementation details.

## What signAI does

signAI is a runtime behavioral integrity monitor for PyTorch models. It helps teams detect suspicious model behavior during:

- inference
- training
- hosted cloud deployments
- self-hosted deployments
- air-gapped deployments

The product is built around one SDK and one server contract. Deployment mode changes where scoring is routed, not how the customer uses the monitor.

## Core Concepts

### Monitor

The SDK object that wraps a customer model and exposes:

- calibration
- artifact save/load
- inference scoring
- training scoring

### Artifact

A compact JSON monitor artifact generated after calibration. It contains detector parameters and thresholds, not model weights.

### Local mode

The artifact is loaded and scored locally. No server is required.

### Remote mode

Behavioral vectors are extracted locally and sent to a configured endpoint for scoring.

## Deployment Modes

### Local artifact mode

Best for:

- offline notebooks
- single-user workflows
- evaluation pilots

### Hosted signAI cloud

Best for:

- fast onboarding
- centralized operations
- lower setup burden

### Self-hosted

Best for:

- VPC deployments
- internal platform teams
- centralized monitor management

### Air-gapped

Best for:

- regulated environments
- isolated infrastructure
- no-internet deployments

## Customer Workflow

1. Attach a monitor to a model
2. Calibrate on clean data
3. Save the artifact
4. Load and score locally or through a configured service endpoint
5. Route flagged results into fallback, quarantine, or alerting flows

## Runtime API Shape

Customers use the same hot-path calls across deployment modes:

```python
result = m.score_inference(x, y)
result = m.score_training(logits, loss)
```

Important training note:

- `score_training` is called after `optimizer.step()`
- it receives `logits` and `loss`
- it does not take raw inputs or the optimizer object

## Privacy Model

Privacy is structural, not just a policy promise.

- feature extraction happens on the customer machine
- only compact behavioral vectors are transmitted in remote mode
- model weights do not leave the customer environment
- raw training samples and raw inputs do not leave the customer environment
- server-side persistence stores scores, flags, timestamps, and phase metadata rather than raw behavioral vectors

## Recommended Rollout

1. Start with local artifact mode for validation
2. Calibrate on representative clean traffic
3. Check thresholds against clean and known-bad samples
4. Move to hosted or self-hosted server mode if centralized history or operations are needed
5. Add notifier and audit capabilities for production environments

## Operations Overview

Typical production deployments include:

- a monitor artifact per model
- a scoring service endpoint for centralized workflows
- optional alerting and event history
- edition-based storage options depending on customer requirements

## Public vs Private Repository Boundary

The public `signai` repository contains:

- overview materials
- selected figures
- public-facing documentation
- customer onboarding copy

The private `signAI-core` repository contains:

- production runtime implementation
- SDK and server code
- packaging and deployment internals
- test suite and release machinery

## Next Step

If you are evaluating signAI commercially or operationally, use this repo as the overview layer and request access or a deployment path for the private production delivery.
