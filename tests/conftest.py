"""********************************************************************************
* Copyright (c) 2026 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************

Pytest configuration that conditionally wraps tests as benchmarks.

Set the ``QRISP_BENCHMARK_ALL`` environment variable to ``true`` to activate
the automatic benchmark wrapping.  This is intended for CI benchmark jobs
that run the full test suite through ``pytest-benchmark`` for regression
detection.

The wrapping is selectively applied:
- Tests with fixture arguments are skipped (they typically use mocks
  that break under benchmark's multiple-execution pattern).
- Tests under ``jax_tests/`` use ``benchmark.pedantic(rounds=1)`` for
  single execution to avoid hangs caused by global state leakage in
  the JASP evaluation pipeline when the same test function is called
  more than once.

Regular test runs (without the env var) are completely unaffected.
"""

import os

BENCHMARK_ALL = os.environ.get("QRISP_BENCHMARK_ALL", "").lower() in ("1", "true", "yes")

if BENCHMARK_ALL:

    def _skip_wrapping(item):
        """Return True if this test item should NOT be benchmark-wrapped.

        Tests that take fixture arguments (likely use mocks or per-test
        state that breaks under benchmark's multiple-execution pattern)
        should not be wrapped: the benchmark wrapper runs the test body
        multiple times, causing mock assertions like ``assert_called_once``
        to fail.
        """
        if item._fixtureinfo.argnames:
            return True
        return False

    def pytest_collection_modifyitems(items):
        """Inject the ``benchmark`` fixture into collected test items."""
        for item in items:
            if "benchmark" in item.fixturenames or _skip_wrapping(item):
                continue
            item.fixturenames = list(item.fixturenames) + ["benchmark"]

    def pytest_pyfunc_call(pyfuncitem):
        """Run plain test functions through the ``benchmark`` fixture."""
        if "benchmark" not in pyfuncitem.fixturenames:
            return

        testfunction = pyfuncitem.obj

        try:
            fnames = pyfuncitem._fixtureinfo.argnames
            fixture_values = {name: pyfuncitem._request.getfixturevalue(name) for name in fnames if name != "benchmark"}
        except Exception:
            return

        try:
            benchmark = pyfuncitem._request.getfixturevalue("benchmark")
        except Exception:
            return

        # Tests under jax_tests/ have global state in the jasp
        # evaluation pipeline (TracingQuantumSession singleton,
        # terminal sampling buffers) that leaks between repeated
        # runs, causing infinite loops in evaluate_while_loop.
        # Use pedantic with a single round to guarantee exactly
        # one execution (benchmark.__call__ runs the function 3x:
        # calibration + measurement + final result).
        if "/jax_tests/" in str(pyfuncitem.path):
            result = benchmark.pedantic(testfunction, kwargs=fixture_values, rounds=1, iterations=1)
        else:
            result = benchmark(testfunction, **fixture_values)
        return result if result is not None else True
