"""The one place database-backed tests get a connection string.

Database tests delete every row they touch on teardown, so they must never reach a database that
holds household data. They read only ``TEST_DATABASE_URL`` (never ``DATABASE_URL``, which the
service itself uses) and refuse any database whose name does not end in ``_test``. Without
``TEST_DATABASE_URL`` the database tests skip.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

TEST_DATABASE_SUFFIX = "_test"


def safe_test_dsn(environ: dict[str, str] | None = None) -> str | None:
    """Return ``TEST_DATABASE_URL``, None when unset; raise if it names a non-test database."""
    dsn = (environ if environ is not None else os.environ).get("TEST_DATABASE_URL", "").strip()
    if not dsn:
        return None
    database = urlparse(dsn).path.lstrip("/")
    if not database.endswith(TEST_DATABASE_SUFFIX):
        raise RuntimeError(
            f"TEST_DATABASE_URL names database {database!r}; database tests delete rows and only "
            f"run against a database whose name ends in {TEST_DATABASE_SUFFIX!r}"
        )
    return dsn
