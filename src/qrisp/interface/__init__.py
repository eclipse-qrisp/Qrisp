"""
********************************************************************************
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

from qrisp.interface.batched_backend import *


# NOTE: IQMBackend is exposed via __getattr__ (PEP 562 lazy module attribute) rather than a
# normal import because of a circular dependency:
#
#   iqm.qrisp_iqm.backends.backend  imports  qrisp.interface  (for Backend, MeasurementResult, Job)
#   qrisp.interface                 imports  iqm.qrisp_iqm.backends  (for IQMBackend)
#
# Attempting a normal import at module-load time causes qrisp.interface to be only
# partially initialised when iqm.qrisp_iqm.backends.backend tries to import from it,
# which raises ImportError for names like Backend and Job that haven't been bound yet.
#
# __getattr__ is only invoked when the caller explicitly requests the name
# (e.g. "from qrisp.interface import IQMBackend"), at which point this module is fully
# initialised and the circular dependency no longer exists.
def __getattr__(name: str):
    if name == "IQMBackend":
        try:
            from iqm.qrisp_iqm.backends import IQMBackend

            return IQMBackend
        except ImportError as exc:
            raise ImportError(
                "IQMBackend requires iqm-client with qrisp support. "
                "Install it with: pip install qrisp[iqm]"
            ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from qrisp.interface.measurement_result import (
    LazyDict,
    MeasurementResult,
    DecodedMeasurementResult,
    MultiMeasurementResult,
)
from qrisp.interface.converter import *
from qrisp.interface.provider_backends import *
from qrisp.interface.backend import *
from qrisp.interface.job import *
from qrisp.interface.simulators import *
from qrisp.interface.virtual_backend import *
