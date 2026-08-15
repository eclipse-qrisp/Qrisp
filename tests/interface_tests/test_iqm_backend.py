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

"""Tests for the IQMBackend import shim.

The actual ``IQMBackend`` implementation lives in the ``iqm.qrisp_iqm``
package and is tested there.  The Qrisp-side module
:mod:`qrisp.interface.provider_backends.iqm_backend` is a thin delegation
shim that:

1. Re-exports ``IQMBackend`` from ``iqm.qrisp_iqm`` when the package is
   installed.
2. Provides a ``_MissingIQMBackend`` placeholder with a helpful
   ``ImportError`` when it is not.

These tests verify **only** the shim layer — not the backend itself.
"""

import pytest

from qrisp.interface import IQMBackend
from qrisp.interface.backend import Backend


class TestShimImport:
    """Verify the re-export and placeholder behave correctly."""

    def test_import_from_interface(self):
        """``IQMBackend`` is importable from ``qrisp.interface``."""
        assert IQMBackend is not None

    def test_iqm_available_flag(self):
        """``_IQM_AVAILABLE`` is ``True`` when the IQM package is installed."""
        from qrisp.interface.provider_backends.iqm_backend import _IQM_AVAILABLE

        assert _IQM_AVAILABLE is True

    def test_is_backend_subclass(self):
        """The re-exported ``IQMBackend`` is a subclass of :class:`~qrisp.interface.Backend`."""
        assert issubclass(IQMBackend, Backend)

    def test_placeholder_import_error(self):
        """The ``_MissingIQMBackend`` placeholder raises ``ImportError`` with install instructions."""
        from qrisp.interface.provider_backends.iqm_backend import _MissingIQMBackend

        with pytest.raises(ImportError, match="pip install qrisp\\[iqm\\]"):
            _MissingIQMBackend()

    def test_placeholder_import_error_with_args(self):
        """The placeholder raises ``ImportError`` even when called with constructor args."""
        from qrisp.interface.provider_backends.iqm_backend import _MissingIQMBackend

        with pytest.raises(ImportError, match="pip install qrisp\\[iqm\\]"):
            _MissingIQMBackend(server_url="http://example.com", device_instance="garnet")
