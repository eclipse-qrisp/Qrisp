import os

BENCHMARK_ALL = os.environ.get("QRISP_BENCHMARK_ALL", "").lower() in ("1", "true", "yes")

if BENCHMARK_ALL:

    def pytest_collection_modifyitems(items):
        for item in items:
            if "benchmark" not in item.fixturenames:
                item.fixturenames = list(item.fixturenames) + ["benchmark"]

    def pytest_pyfunc_call(pyfuncitem):
        testfunction = pyfuncitem.obj

        try:
            fnames = pyfuncitem._fixtureinfo.argnames
            fixture_values = {name: pyfuncitem._request.getfixturevalue(name) for name in fnames}
        except Exception:
            return

        try:
            benchmark = pyfuncitem._request.getfixturevalue("benchmark")
        except Exception:
            return

        result = benchmark(testfunction, **fixture_values)
        return result if result is not None else True
