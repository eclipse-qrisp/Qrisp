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

# ====================================================================== #
# qrisp.jasp.cudaq_interface.annotations
# Type annotations for @cudaq_kernel parameters.
# ====================================================================== #

import numpy as np


class FixedShapeNDArray:
    """Type annotation for a fixed-size numpy array parameter in
    ``@cudaq_kernel``.

    Mirrors the role of ``list[float]`` / ``list[int]`` in ``@cudaq.kernel``
    but requires an explicit size, because JAX traces with concrete
    shapes. The decorator uses the element type and size to generate a
    correctly-typed dummy value for tracing.

    Parameters
    ----------
    dtype : type
        Element type.  Supported: ``float``, ``int``, ``bool``.
    size : int
        Number of elements.  Must match the runtime array length.

    Examples
    --------
    ::

        import cudaq
        import numpy as np
        from qrisp import *
        from qrisp.jasp.cudaq_interface import cudaq_kernel, FixedShapeNDArray

        @cudaq_kernel
        def circuit(angles: FixedShapeNDArray(float, 3)):
            qv = QuantumFloat(2)
            ry(angles[0], qv[0])
            return measure(qv[0])

        angles = np.array([1.57, 0.78, 0.39])
        print(cudaq.run(circuit, angles, shots_count=100))

    """

    #: Maps Python dtype to NumPy dtype and MLIR element type string.
    _DTYPE_MAP = {
        float: (np.float64, "f64"),
        int: (np.int64, "i64"),
        bool: (np.bool_, "i1"),
    }

    def __init__(self, dtype: type, size: int):
        if dtype not in self._DTYPE_MAP:
            raise TypeError(
                f"FixedShapeNDArray: unsupported dtype '{dtype}'. Supported: {list(self._DTYPE_MAP.keys())}."
            )
        if not isinstance(size, int) or size <= 0:
            raise ValueError("FixedShapeNDArray: size must be a positive integer.")
        self.dtype = dtype
        self.size = size
        self._np_dtype, self.mlir_elem_type = self._DTYPE_MAP[dtype]

    def make_dummy(self) -> np.ndarray:
        """Return a zero-filled NumPy array of the correct dtype and size."""
        return np.zeros(self.size, dtype=self._np_dtype)

    def __repr__(self):
        return f"FixedShapeNDArray({self.dtype.__name__}, {self.size})"
