-- Refuses while attested rows exist: they are household face crops with audit history, and the
-- previous schema cannot hold them. Delete them deliberately (or keep this version) first.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM calibration_candidates WHERE label_state = 'attested') THEN
        RAISE EXCEPTION 'calibration_candidates holds attested rows; delete them before rolling back';
    END IF;
END
$$;

DROP INDEX IF EXISTS idx_cal_cand_attested_audit;
ALTER TABLE calibration_candidates DROP CONSTRAINT IF EXISTS calibration_candidates_attestation_complete;
ALTER TABLE calibration_candidates DROP CONSTRAINT calibration_candidates_label_complete;
ALTER TABLE calibration_candidates ADD CONSTRAINT calibration_candidates_label_complete CHECK (
    (label_state = 'labeled') = (label_kind IS NOT NULL)
    AND (label_kind IS NULL OR (label_source IS NOT NULL AND labeled_by IS NOT NULL
                                AND labeled_at IS NOT NULL))
    AND (COALESCE(label_kind = 'member', false) = (label_identity_id IS NOT NULL))
);
ALTER TABLE calibration_candidates
    DROP COLUMN IF EXISTS audited_at,
    DROP COLUMN IF EXISTS audited_by,
    DROP COLUMN IF EXISTS audit_verdict,
    DROP COLUMN IF EXISTS attestation_grade,
    DROP COLUMN IF EXISTS attestation_association_id;
ALTER TABLE calibration_candidates
    DROP CONSTRAINT IF EXISTS calibration_candidates_label_state_check;
ALTER TABLE calibration_candidates
    ADD CONSTRAINT calibration_candidates_label_state_check
    CHECK (label_state IN ('unlabeled', 'labeled', 'rejected'));
