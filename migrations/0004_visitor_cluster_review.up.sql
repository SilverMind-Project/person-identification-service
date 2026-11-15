-- Visitor cluster review: attribute a cluster to an enrolled member without enrolling it, and
-- record who reviewed a cluster and when.
--
-- 'attributed' closes a cluster as faces of an existing member. It never adds embeddings to the
-- member: enrollment changes only through explicit enrollment or naming. The foreign key has no
-- delete action on purpose: deleting a member reverts its attributed clusters first (in the same
-- transaction), so a code path that forgets to fails loudly instead of orphaning the label.
ALTER TABLE visitor_clusters
    ADD COLUMN IF NOT EXISTS attributed_person_id TEXT NULL REFERENCES members(person_id),
    ADD COLUMN IF NOT EXISTS reviewed_by TEXT NULL,
    ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ NULL;

ALTER TABLE visitor_clusters DROP CONSTRAINT IF EXISTS visitor_clusters_status_check;
ALTER TABLE visitor_clusters ADD CONSTRAINT visitor_clusters_status_check
    CHECK (status IN ('candidate', 'surfaced', 'named', 'dismissed', 'attributed'));

-- An attributed cluster always names its member, and only an attributed cluster does.
ALTER TABLE visitor_clusters DROP CONSTRAINT IF EXISTS visitor_clusters_attribution_check;
ALTER TABLE visitor_clusters ADD CONSTRAINT visitor_clusters_attribution_check
    CHECK ((status = 'attributed') = (attributed_person_id IS NOT NULL));

CREATE INDEX IF NOT EXISTS visitor_clusters_attributed_person_idx
    ON visitor_clusters (attributed_person_id) WHERE attributed_person_id IS NOT NULL;
