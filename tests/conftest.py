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

Pytest configuration that conditionally wraps every test as a benchmark.

Set the ``QRISP_BENCHMARK_ALL`` environment variable to ``true`` to activate
the automatic benchmark wrapping.  This is intended for CI benchmark jobs
that run the full test suite through ``pytest-benchmark`` for regression
detection.

Regular test runs (without the env var) are completely unaffected.
"""

import os

BENCHMARK_ALL = os.environ.get("QRISP_BENCHMARK_ALL", "").lower() in ("1", "true", "yes")

if BENCHMARK_ALL:

    def pytest_collection_modifyitems(items):
        """Inject the ``benchmark`` fixture into every collected test item."""
        for item in items:
            if "benchmark" not in item.fixturenames:
                item.fixturenames = list(item.fixturenames) + ["benchmark"]

    def pytest_pyfunc_call(pyfuncitem):
        """Run every test function through the ``benchmark`` fixture.

        Resolves fixture arguments, wraps the call with
        :meth:`pytest_benchmark.fixture.benchmark`, and returns the
        result so pytest treats it as a normal pass/fail.
        """
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
