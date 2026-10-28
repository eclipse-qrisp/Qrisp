# ********************************************************************************
# * Copyright (c) 2026 the Qrisp authors
# *
# * This program and the accompanying materials are made available under the
# * terms of the Eclipse Public License 2.0 which is available at
# * http://www.eclipse.org/legal/epl-2.0.
# *
# * This Source Code may also be made available under the following Secondary
# * Licenses when the conditions for such availability set forth in the Eclipse
# * Public License, v. 2.0 are satisfied: GNU General Public License, version 2
# * with the GNU Classpath Exception which is
# * available at https://www.gnu.org/software/classpath/license.html.
# *
# * SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
# ********************************************************************************

"""Re-exports IQMBackend from the optional iqm package, raising ImportError with install hint if missing."""


class _MissingIQMBackend:
    """Placeholder for :class:`IQMBackend` when the ``iqm`` package is not installed.

    Attempting to instantiate this placeholder raises an :exc:`ImportError`
    with instructions to install ``qrisp[iqm]``.
    """

    def __new__(cls, *args, **kwargs):
        raise ImportError(
            "The IQM backend requires the 'iqm' package, which is not installed.\n"
            "Install it with: pip install qrisp[iqm]"
        )


try:
    from iqm.qrisp_iqm import IQMBackend

    _IQM_AVAILABLE = True
except ImportError:
    _IQM_AVAILABLE = False
    IQMBackend = _MissingIQMBackend

__all__ = ["IQMBackend"]
