# signAI

Public front for the private `signAI-core` product.

This repository is the public-facing overview for signAI. It contains:

- the public landing page
- selected result figures
- customer-facing overview docs
- public manual and deployment overview

The full product implementation lives in the private `signAI-core` repository.

## What signAI is

signAI is a runtime behavioral integrity monitor for PyTorch models.

It models the relationship between:

- an input-derived or update-conditioned signature `S`
- an internal behavior signature `Z`

and scores whether the observed model behavior is consistent with the calibrated baseline.

## Public materials in this repo

- `index.html` - public landing page
- `USER_MANUAL.md` - customer-facing manual
- `DEPLOY.md` - public deployment overview
- `figures/public_results_snapshot.png` - summary figure
- `figures/roc_i1_svhn.png` - signAI I1 ROC
- `figures/roc_mahal_svhn.png` - Mahalanobis baseline ROC
- `figures/roc_energy_svhn.png` - energy baseline ROC

## Repository boundary

### Public `signai` repo

Use this repo to:

- understand the product concept
- review public figures
- share the landing page
- onboard early customer conversations

### Private `signAI-core` repo

The private repo contains:

- the SDK
- the server runtime
- calibration and scoring implementation
- deployment packaging
- tests and release machinery

## Core idea

signAI treats internal model behavior as something that can be monitored at runtime, not only explained after the fact.

If internal behavior is structured enough to model conditionally from inputs or updates, then that structure can become a live integrity signal.

## Access

The complete system and operational delivery are maintained privately.

For access or customer deployment discussion, contact:

- `umarjanjua@live.com`
