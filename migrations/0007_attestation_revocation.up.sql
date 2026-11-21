-- Revocation of wearable-attested face exemplars by association segment.
-- Revoked rows stay excluded from the audit sample, the calibration manifest, and promotion,
-- and purge on the same schedule as unaudited captures. Already-promoted face prototypes are
-- never removed automatically: they are returned to the operator for manual review.
ALTER TABLE calibration_candidates
    ADD COLUMN IF NOT EXISTS revoked_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS revoked_by TEXT;

ALTER TABLE calibration_candidates
    ADD CONSTRAINT calibration_candidates_revocation_complete CHECK (
        (revoked_at IS NULL) = (revoked_by IS NULL)
        AND (revoked_at IS NULL OR label_state = 'attested')
    );

CREATE INDEX IF NOT EXISTS idx_cal_cand_attested_association
    ON calibration_candidates (attestation_association_id)
    WHERE label_state = 'attested';
