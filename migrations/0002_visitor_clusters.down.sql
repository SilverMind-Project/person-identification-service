-- Rollback for 0002_visitor_clusters.
-- NOTE: the migration runner (app/db/migrate.py) only scans *.up.sql and has
-- no automatic down-migration path. This file is applied manually (e.g. via
-- psql) if a full rollback is needed; delete the visitor-crops/ MinIO prefix
-- separately. Enrollment members created by naming a cluster are NOT touched
-- (they are real members; removing one is the standard remove_member path).
DROP TABLE IF EXISTS visitor_sightings;
DROP TABLE IF EXISTS visitor_clusters;
