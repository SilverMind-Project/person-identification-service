"""Database tests can only ever reach a database named for tests."""

from __future__ import annotations

import pytest

from tests.db import safe_test_dsn


def test_unset_means_no_database() -> None:
    assert safe_test_dsn({}) is None
    assert safe_test_dsn({"TEST_DATABASE_URL": "  "}) is None


def test_service_database_url_is_never_used() -> None:
    live = "postgresql://u:p@localhost:5432/person_identification"
    assert safe_test_dsn({"DATABASE_URL": live}) is None


@pytest.mark.parametrize(
    "database", ["person_identification", "person_identification_testing", "test", "postgres"]
)
def test_a_non_test_database_is_refused(database: str) -> None:
    with pytest.raises(RuntimeError, match="_test"):
        safe_test_dsn({"TEST_DATABASE_URL": f"postgresql://u:p@localhost:5432/{database}"})


def test_a_test_database_is_returned() -> None:
    dsn = "postgresql://u:p@localhost:5432/person_identification_test"
    assert safe_test_dsn({"TEST_DATABASE_URL": dsn}) == dsn
