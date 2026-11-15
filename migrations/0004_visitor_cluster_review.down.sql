-- Attributed clusters become dismissed (the closest earlier state: closed, absorbing, never
-- surfaced) before the attribution columns and the widened status check are removed.
UPDATE visitor_clusters SET status = 'dismissed', attributed_person_id = NULL
    WHERE status = 'attributed';

DROP INDEX IF EXISTS visitor_clusters_attributed_person_idx;
ALTER TABLE visitor_clusters DROP CONSTRAINT IF EXISTS visitor_clusters_attribution_check;
ALTER TABLE visitor_clusters DROP CONSTRAINT IF EXISTS visitor_clusters_status_check;
ALTER TABLE visitor_clusters ADD CONSTRAINT visitor_clusters_status_check
    CHECK (status IN ('candidate', 'surfaced', 'named', 'dismissed'));

ALTER TABLE visitor_clusters
    DROP COLUMN IF EXISTS reviewed_at,
    DROP COLUMN IF EXISTS reviewed_by,
    DROP COLUMN IF EXISTS attributed_person_id;
