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
"""

import subprocess
import sys


def test_qrisp_import_does_not_load_optional_converter_libraries():
    """Importing Qrisp does not import the libraries used by the circuit converters."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from qrisp import *

for library in ("pennylane", "pytket", "qulacs", "stim", "cirq"):
    assert library not in sys.modules, library
""",
        ],
        check=True,
    )
