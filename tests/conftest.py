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

Set ``QRISP_BENCHMARK_ALL=true`` to activate.  Intended for CI benchmark jobs
that run the full test suite through ``pytest-benchmark`` for regression
detection.

Tests with fixture arguments are left alone (their mocks would break under
benchmark's multiple-execution pattern).  Tests under ``jax_tests/`` use
``pedantic(rounds=1)`` to avoid hangs caused by global state leakage in the
JASP evaluation pipeline.
"""

import os

BENCHMARK_ALL = os.environ.get("QRISP_BENCHMARK_ALL", "").lower() in ("1", "true", "yes")

if BENCHMARK_ALL:

    def pytest_collection_modifyitems(items):
        for item in items:
            if item._fixtureinfo.argnames:
                continue
            if "benchmark" not in item.fixturenames:
                item.fixturenames = list(item.fixturenames) + ["benchmark"]

    def pytest_pyfunc_call(pyfuncitem):
        if "benchmark" not in pyfuncitem.fixturenames:
            return

        try:
            benchmark = pyfuncitem._request.getfixturevalue("benchmark")
        except Exception:
            return

        if "/jax_tests/" in str(pyfuncitem.path):
            result = benchmark.pedantic(pyfuncitem.obj, rounds=1, iterations=1)
        else:
            result = benchmark(pyfuncitem.obj)
        return result if result is not None else True
