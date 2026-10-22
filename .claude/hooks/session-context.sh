#!/usr/bin/env bash
# person-identification-service: SessionStart context (startup and after compaction), at most 10 lines (EG-M06).
set -u
cd "${CLAUDE_PROJECT_DIR:-$(pwd)}" 2>/dev/null || exit 0
branch=$(git branch --show-current 2>/dev/null)
dirty=$(git status --short 2>/dev/null | wc -l)
echo "[person-identification-service] branch ${branch:-?}, ${dirty// /} files changed or untracked."
echo "Gate: TEST_DATABASE_URL=postgresql://pid_user:change-me-pid-password@localhost:5432/person_identification_test PID_REQUIRE_FULL_SUITE=1 make check"
echo "Never run tests against the live person_identification database; record the live write counter before and after:"
echo "  docker exec postgres psql -U postgres -d person_identification -At -c \"select sum(n_tup_ins)+sum(n_tup_del) from pg_stat_user_tables\""
echo "The live gallery may be empty until residents are re-enrolled."
echo "Guards: .claude/hooks/ (project) and ~/.claude/hooks/guard-machine.sh (machine). Override with ! if intended."
