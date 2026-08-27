"""Temporary test to verify the mypy reviewdog check fails as expected."""


def _type_check_failure(x: int) -> str:
    """Temporary helper used to verify the mypy reviewdog check fails."""
    return x
