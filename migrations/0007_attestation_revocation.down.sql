-- Refuse rollback while revoked attested captures exist so revocation audit state is never
-- silently discarded.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM calibration_candidates WHERE revoked_at IS NOT NULL) THEN
        RAISE EXCEPTION 'Cannot roll back 0007_attestation_revocation while revoked rows exist';
    END IF;
END $$;

DROP INDEX IF EXISTS idx_cal_cand_attested_association;

ALTER TABLE calibration_candidates
    DROP CONSTRAINT IF EXISTS calibration_candidates_revocation_complete;

ALTER TABLE calibration_candidates
    DROP COLUMN IF EXISTS revoked_by,
    DROP COLUMN IF EXISTS revoked_at;
