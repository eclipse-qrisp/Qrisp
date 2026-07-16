import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "benchmark" not in item.fixturenames:
            item.fixturenames = list(item.fixturenames) + ["benchmark"]


def pytest_pyfunc_call(pyfuncitem):
    testfunction = pyfuncitem.obj

    try:
        fixture_values = {name: pyfuncitem._request.getfixturevalue(name) for name in pyfuncitem._fixtureinfo.argnames}
    except Exception:
        return

    try:
        benchmark = pyfuncitem._request.getfixturevalue("benchmark")
    except Exception:
        return

    result = benchmark(testfunction, **fixture_values)
    return result if result is not None else True
