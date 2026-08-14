# signAI Deployment Overview

## Public Note

This document is the public deployment overview for signAI.

The full operational implementation lives in the private `signAI-core` repository. This page exists so customers can understand deployment options before receiving the private delivery.

## Deployment Options

### Local artifact scoring

Scoring runs in-process against a saved artifact — no server, no network.

Use when:

- a team wants no service dependency at inference time
- a pilot is running on a single machine
- an environment is fully offline

Calibration is separate: producing an artifact requires the local signAI daemon
(`signai serve`, on `localhost:7731`). Once the artifact exists the daemon can be
stopped and scoring continues without it.

### Self-hosted mode

Use when:

- the service must run in a customer VPC
- the customer wants to own storage and alerting integration
- internal platform teams need a shared scoring service

### Air-gapped mode

Use when:

- infrastructure has no internet access
- environments are regulated or isolated
- the same server contract must run entirely inside private networking

## Common Architecture

1. Customer model runs locally
2. Behavioral vectors are extracted locally
3. Scoring runs either:
   - in-process from a local artifact, or
   - via a self-hosted endpoint you operate
4. Production workflows consume score and flag results

## Privacy Boundary

- model weights stay in the customer environment
- raw inputs stay in the customer environment
- remote mode transmits compact behavioral vectors only
- stored history contains score metadata rather than raw feature vectors

## Typical Customer Rollout

1. Pilot locally
2. Validate calibration quality
3. Choose local-artifact or self-hosted routing
4. Add alerting, history, and audit features as needed
5. Move into production operations

## Customer Delivery

Production delivery is typically handled through the private `signAI-core` track, which contains:

- the product SDK
- the scoring server
- packaging details
- deployment automation
- testing and release support

## Contact / Access

Use the public repo for overview and discussion, then move into the private delivery path for actual implementation and deployment.
