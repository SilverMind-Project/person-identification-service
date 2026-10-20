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

The deprecated face `confidence` field (a mirror of raw similarity) was removed on 2026-10-20;
`tests/test_identification_geometry_contract.py` and
`tests/test_confidence_contract_characterization.py` assert it stays absent. `similarity` is the
documented match score; `raw_similarity` remains a computed alias of it for one release (see the
engineering-standards skill's deprecation note).

Missing, invalid, corrupt, or incompatible calibration produces `calibrated_confidence=null` and a
degraded health state. It never falls back to raw similarity.

## Face quality

Every identified face carries `quality {score, resolution, sharpness, detection, pitch, version}`,
all in [0, 1], or `null` for a face that was not scored.

- Computed only in `app/services/face_quality.py` behind the `FaceQualityScorer` protocol, from
  `FaceQualityInputs` (interocular distance from `face_geometry()`, SCRFD `det_score`, head
  pitch, and the grayscale ArcFace-aligned crop), inside `FaceEngine._enrich_face` so alignment
  runs once. The scorer is built in the lifespan from `face_quality` settings; an unknown
  `scorer` name or key fails startup before any connection opens.
- `heuristic-v1`: clamped linear ramps combined by a weighted geometric mean (resolution 0.45,
  sharpness 0.25, detection 0.15, pitch 0.15; each component floored at 0.01). Yaw is excluded
  because the tracker applies its own frontality factor. Missing interocular distance scores
  resolution 0, never high.
- The version string is the contract. Any change to anchors, weights, components, or the
  aligned-crop preprocessing is a new version; a calibration artifact or downstream policy
  conditioned on quality binds to the version it was fitted on and must be re-checked when it
  changes.
- Anchors are tuned offline with `scripts/calibrate_face_quality.py` (operator-labeled images,
  `FaceEngine` directly, read-only centroid query, never writes config) and the pure analysis in
  `app/calibration/quality_breakpoints.py` (scipy, `calibration-tools` extra only).

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

