-- Drops every calibration capture, labeled ones included. Export labels first if they should be
-- kept; the MinIO crops under calibration/ are not removed by this file.
DROP TABLE IF EXISTS calibration_candidates;
