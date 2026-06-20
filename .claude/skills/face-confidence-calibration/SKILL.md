---
name: face-confidence-calibration
description: Use when changing ArcFace similarity outputs, confidence calibration, recognition-state contracts, calibration datasets, artifacts, or person-identification health reporting.
---

# Face Confidence Calibration

Load this skill together with:

- `/home/sriram/code/nanai/person-identification-service/AGENTS.md`
- `/home/sriram/code/nanai/continuous-tracking/.claude/skills/cts-identity-governance/SKILL.md` when changing CTS consumers

## Terminology

- `raw_similarity`: cosine similarity between normalized ArcFace embeddings.
- `calibrated_confidence`: estimated probability of a same-identity match produced by a validated,
  version-compatible calibration artifact.
- `recognition_state`: compatibility state (`recognized`, `candidate`, `unrecognized`) derived from
  configured raw-similarity thresholds.
- `authoritative`: downstream policy decision requiring calibrated confidence, direct evidence,
  unambiguous face/body association, and compatible versions.

Never name cosine similarity `confidence`, `probability`, or `certainty`.

## Service contract

Identification responses expose both `raw_similarity` and nullable `calibrated_confidence`, plus:

- calibration status and artifact version;
- ArcFace model/profile version;
- preprocessing version;
- recognition state;
- face detection confidence separately.

During compatibility migration, a deprecated `confidence` field may mirror raw similarity only if
documented and never consumed as calibrated confidence. Add contract tests and remove it in a
dedicated cleanup milestone.

Missing, invalid, corrupt, or incompatible calibration produces `calibrated_confidence=null` and a
degraded health state. It never falls back to raw similarity.

## Calibration dataset

Calibration requires labeled same-person and different-person face pairs. Data rules:

- use explicit labels, never inferred PH or tracking labels;
- split by identity and capture episode to prevent adjacent-frame leakage;
- keep fitting, threshold-selection, and final evaluation partitions separate;
- do not fit solely on one household or two identities;
- household data may be a separate holdout for validation;
- ordinary identity correction does not automatically add a sample;
- an unambiguous reviewed face may be explicitly added to a calibration-candidate set without
  changing ArcFace enrollment.

Household images, crops, embeddings, and populated manifests are local/private artifacts and must
be ignored by Git. Commit only schemas, synthetic examples, loaders, and tests.

## Tooling

Use established open-source calibration implementations offline:

- scikit-learn logistic regression for Platt-style calibration;
- scikit-learn isotonic regression when sample size and coverage justify it;
- scikit-learn metrics for Brier score, log loss, ROC/DET, false-match, and false-nonmatch rates.

Do not load pickle/joblib artifacts in the service. Export a validated JSON artifact containing
typed coefficients or isotonic knots, metadata, dataset hashes, fit metrics, and compatibility
versions. Runtime evaluation uses a small deterministic typed loader.

The fitter must:

1. validate manifest schema and file hashes;
2. reject identity/episode leakage;
3. compare candidate calibrators against an uncalibrated baseline;
4. report reliability bins and operating-point metrics;
5. write an artifact only when configured acceptance gates pass;
6. produce deterministic output for a fixed seed and dataset;
7. never mutate enrollment or production data.

## Artifact contract

Required metadata:

- schema version;
- artifact ID and creation time;
- ArcFace model/profile and preprocessing versions;
- calibrator type and parameters;
- fit/validation/test dataset hashes and counts;
- identity and episode counts per split;
- Brier score, log loss, ECE/reliability summary;
- false-match and false-nonmatch rates at declared operating points;
- tool version and random seed.

Startup validates the artifact before serving calibrated values. Health exposes `ready`,
`degraded_missing`, `degraded_incompatible`, or `degraded_invalid` with a non-secret reason.

## Required tests

- raw similarity and calibrated confidence remain distinct in models and JSON;
- runtime logistic and isotonic evaluators match scikit-learn fixtures;
- missing/incompatible/corrupt artifacts fail closed;
- recognition state remains compatible with raw thresholds;
- model or preprocessing version changes invalidate old artifacts;
- split validator detects identity and capture-episode leakage;
- deterministic fitter output for a fixed synthetic dataset;
- artifact acceptance gates reject underperforming calibration;
- health endpoint reports every calibration state;
- CTS consumer contract test proves raw similarity cannot satisfy authority.

## Review checklist

- [ ] No inferred labels entered the dataset.
- [ ] Fit and evaluation data are disjoint by identity and episode as configured.
- [ ] Runtime has no scikit-learn or pickle requirement.
- [ ] Missing calibration returns null, not a heuristic substitute.
- [ ] Every artifact is version-compatible and reproducible.
- [ ] API and health changes are tested in person-identification and CTS.

