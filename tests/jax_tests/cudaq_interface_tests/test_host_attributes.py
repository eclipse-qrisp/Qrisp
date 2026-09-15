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

import pytest

from qrisp.jasp.cudaq_interface.cudaq_ingestion import host_attributes

# ---------------------------------------------------------------------------
# Test _detect_platform_key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sys_platform, expected_os_key",
    [("linux", "linux"), ("linux2", "linux"), ("darwin", "darwin"), ("win32", "win32")],
)
def test_detect_platform_key(monkeypatch, sys_platform, expected_os_key):
    """_detect_platform_key normalizes sys.platform into a (machine, os_key) pair."""
    monkeypatch.setattr(host_attributes.sys, "platform", sys_platform)
    machine, os_key = host_attributes._detect_platform_key()
    assert machine == host_attributes.platform.machine().lower()
    assert os_key == expected_os_key


# ---------------------------------------------------------------------------
# Test _get_llvm_attributes
# ---------------------------------------------------------------------------


def test_get_llvm_attributes_extracts_target_triple(monkeypatch):
    """Both llvm.data_layout and llvm.target_triple are extracted via regex when present."""

    class _FakeKernel:
        def __str__(self):
            return 'module attributes {llvm.data_layout = "DL-STRING", llvm.target_triple = "TT-STRING"} {}'

    monkeypatch.setattr(host_attributes.cudaq, "kernel", lambda func: _FakeKernel())

    data_layout, target_triple = host_attributes._get_llvm_attributes()
    assert data_layout == "DL-STRING"
    assert target_triple == "TT-STRING"


def test_get_llvm_attributes_falls_back_on_extraction_failure(monkeypatch):
    """If compiling/parsing the dummy kernel raises, the platform-default is used (with a warning)."""

    def _raise(func):
        raise RuntimeError("cudaq unavailable")

    monkeypatch.setattr(host_attributes.cudaq, "kernel", _raise)
    monkeypatch.setattr(host_attributes, "_detect_platform_key", lambda: ("x86_64", "linux"))

    with pytest.warns(UserWarning, match="platform default"):
        result = host_attributes._get_llvm_attributes()

    assert result == host_attributes._PLATFORM_DEFAULTS[("x86_64", "linux")]


def test_get_llvm_attributes_unsupported_platform_raises(monkeypatch):
    """An unsupported (machine, os) key with no extractable MLIR raises RuntimeError."""

    def _raise(func):
        raise RuntimeError("cudaq unavailable")

    monkeypatch.setattr(host_attributes.cudaq, "kernel", _raise)
    monkeypatch.setattr(host_attributes, "_detect_platform_key", lambda: ("riscv64", "plan9"))

    with pytest.raises(RuntimeError, match="no default for"):
        host_attributes._get_llvm_attributes()
