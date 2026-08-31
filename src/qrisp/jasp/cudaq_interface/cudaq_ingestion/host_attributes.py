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

"""Provide host LLVM target attributes required by CUDA-Q."""

# Host/LLVM target attribute detection (pipeline stage 1 – "Target Extraction").
# ================================================================================

import platform
import re
import sys
import warnings

import cudaq

# ------------------------------------------------------------------ #
# Platform-aware LLVM attribute defaults
# ------------------------------------------------------------------ #

_PLATFORM_DEFAULTS = {
    ("x86_64", "linux"): (
        "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
        "x86_64-unknown-linux-gnu",
    ),
    ("aarch64", "linux"): (
        "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128",
        "aarch64-unknown-linux-gnu",
    ),
    ("arm64", "darwin"): (
        "e-m:o-i64:64-i128:128-n32:64-S128-Fn32",
        "arm64-apple-macosx14.0.0",
    ),
    ("x86_64", "darwin"): (
        "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
        "x86_64-apple-macosx14.0.0",
    ),
}


def _detect_platform_key() -> tuple:
    """Return ``(machine, os_prefix)`` for the current host."""
    machine = platform.machine().lower()
    if sys.platform.startswith("linux"):
        os_key = "linux"
    elif sys.platform == "darwin":
        os_key = "darwin"
    else:
        os_key = sys.platform
    return (machine, os_key)


def _get_llvm_attributes() -> tuple[str, str | None]:
    """Extract ``llvm.data_layout`` and ``llvm.target_triple`` strings.

    Strategy:
    1. Compile a dummy ``@cudaq.kernel`` and regex-search its MLIR string.
    2. If that fails, fall back to platform-based defaults.

    Returns
    -------
    (data_layout_attr, target_triple_attr)
        Each is a string like ``'llvm.data_layout = "..."'`` ready for
        insertion into a ``module attributes { }`` block.  The target
        triple may be ``None`` if not available.

    Raises
    ------
    RuntimeError
        If neither extraction nor platform defaults succeed.

    """
    data_layout_str = None
    target_triple_str = None

    try:

        @cudaq.kernel
        def _dummy_extractor():
            pass

        dummy_mlir = str(_dummy_extractor)
        dl_match = re.search(r'llvm\.data_layout\s*=\s*"([^"]+)"', dummy_mlir)
        tt_match = re.search(r'llvm\.target_triple\s*=\s*"([^"]+)"', dummy_mlir)
        if dl_match:
            data_layout_str = dl_match.group(1)
        if tt_match:
            target_triple_str = tt_match.group(1)
    except Exception:
        pass

    if data_layout_str is None:
        key = _detect_platform_key()
        defaults = _PLATFORM_DEFAULTS.get(key)
        if defaults is None:
            raise RuntimeError(
                f"Failed to extract llvm.data_layout from CUDA-Q, "
                f"no default for {key}. Supported: {list(_PLATFORM_DEFAULTS.keys())}"
            )
        data_layout_str, target_triple_str = defaults
        warnings.warn(
            f"Could not extract llvm.data_layout from CUDA-Q; using platform default for {key}.",
            stacklevel=3,
        )

    return data_layout_str, target_triple_str
