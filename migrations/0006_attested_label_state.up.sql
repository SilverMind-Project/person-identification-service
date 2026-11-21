-- Wearable-attested face exemplars. A face held in memory from /identify becomes a calibration
-- candidate in state 'attested' when tracking-orchestrator reports that a wearable association
-- placed a household member on that detection. The label is a machine label (label_source
-- 'sensor_attested', labeled_by 'wearable:<association id>'): it never reaches enrollment, and
-- calibration reads it only once an audited random sample shows 99% precision or better.
ALTER TABLE calibration_candidates
    DROP CONSTRAINT IF EXISTS calibration_candidates_label_state_check;
ALTER TABLE calibration_candidates
    ADD CONSTRAINT calibration_candidates_label_state_check
    CHECK (label_state IN ('unlabeled', 'labeled', 'rejected', 'attested'));

ALTER TABLE calibration_candidates
    ADD COLUMN IF NOT EXISTS attestation_association_id TEXT,
    ADD COLUMN IF NOT EXISTS attestation_grade TEXT
        CHECK (attestation_grade IN ('room', 'trajectory')),
    ADD COLUMN IF NOT EXISTS audit_verdict TEXT
        CHECK (audit_verdict IN ('correct', 'wrong', 'not_a_face')),
    ADD COLUMN IF NOT EXISTS audited_by TEXT,
    ADD COLUMN IF NOT EXISTS audited_at TIMESTAMPTZ;

-- A labeled or attested row names its kind, source, labeler and time; only a member label names
-- an identity. An attested row is always a sensor-attested member label with its association and
-- grade; an audit verdict exists only on an attested row and carries its auditor and time.
ALTER TABLE calibration_candidates DROP CONSTRAINT calibration_candidates_label_complete;
ALTER TABLE calibration_candidates ADD CONSTRAINT calibration_candidates_label_complete CHECK (
    (label_state IN ('labeled', 'attested')) = (label_kind IS NOT NULL)
    AND (label_kind IS NULL OR (label_source IS NOT NULL AND labeled_by IS NOT NULL
                                AND labeled_at IS NOT NULL))
    AND (COALESCE(label_kind = 'member', false) = (label_identity_id IS NOT NULL))
);
ALTER TABLE calibration_candidates ADD CONSTRAINT calibration_candidates_attestation_complete
CHECK (
    (label_state = 'attested') = (attestation_grade IS NOT NULL)
    AND (label_state <> 'attested' OR (label_kind = 'member'
                                       AND label_source = 'sensor_attested'
                                       AND attestation_association_id IS NOT NULL))
    AND (audit_verdict IS NULL OR label_state = 'attested')
    AND ((audit_verdict IS NULL) = (audited_by IS NULL))
    AND ((audit_verdict IS NULL) = (audited_at IS NULL))
);

CREATE INDEX IF NOT EXISTS idx_cal_cand_attested_audit
    ON calibration_candidates (audit_verdict, captured_at DESC)
    WHERE label_state = 'attested';
