.. _GMStools:

GMS Tools
=========

Qrisp provides tools for working with Generalized Middel Swap (GMS) gates.
These include circuit-level conversion, environment-based compilation, and direct gate primitives.

GZZ Converter
-------------

The :meth:`GZZ_converter <qrisp.misc.GMS_tools.GZZ_converter>` function converts a phase-only circuit into a GMS-compatible form while preserving the unitary:

>>> import numpy as np
>>> from qrisp import QuantumVariable, cp, rz
>>> from qrisp.misc.GMS_tools import GZZ_converter
>>> qv = QuantumVariable(5)
>>> cp(np.pi / 4, qv[0], qv[1])
>>> rz(np.pi / 4, qv[0])
>>> qc = GZZ_converter(qv.qs)
>>> print(qv.qs.compare_unitary(qc, precision = 4))
True
>>> print(np.linalg.norm(qc.get_unitary(4) - np.diag(np.diagonal(qc.get_unitary(4)))) < 1e-3)
True

GMSEnvironment
--------------

The :class:`GMSEnvironment <qrisp.environments.GMSEnvironment>` wraps gates for GMS-compatible compilation. Use the ``with`` statement to group operations:

>>> import numpy as np
>>> from qrisp import QuantumVariable, cp, h, p, x
>>> from qrisp.environments import GMSEnvironment
>>> qv1 = QuantumVariable(1)
>>> qv2 = QuantumVariable(1)
>>> h(qv1)
>>> x(qv2)
>>> test = GMSEnvironment()
>>> with test:
...     cp(np.pi / 2, qv1[0], qv2[0])
...     p(np.pi / 2, qv1[0])
...     p(np.pi / 2, qv2[0])
>>> h(qv1)
>>> print(qv1.get_measurement())
{'1': 1.0}

QFT can also be compiled with GMS gates using ``use_gms=True``:

>>> from qrisp import QFT
>>> from qrisp.alg_primitives.arithmetic import QuantumFloat
>>> qf = QuantumFloat(3)
>>> qf.encode(1)
>>> QFT(qf, use_gms = True)

GMS Gate Primitives
-------------------

Low-level GMS gate functions are available in ``qrisp.misc.GMS_tools``:

>>> import numpy as np
>>> from qrisp import QuantumVariable, cp, h
>>> from qrisp.misc.GMS_tools import gms_multi_cp_gate, gms_multi_cx_fan_out
>>> n = 3
>>> theta = [np.pi / (i + 1) for i in range(n - 1)]
>>> qv_0 = QuantumVariable(n)
>>> qv_0.qs.append(gms_multi_cp_gate(n - 1, theta), qv_0.reg)
>>> qv_1 = QuantumVariable(n)
>>> for i in range(n - 1):
...     cp(theta[i], qv_1[i], qv_1[n - 1])
>>> print(qv_0.qs.compare_unitary(qv_1.qs))
True
>>> print(np.linalg.norm(qv_0.qs.get_unitary() - np.diag(np.diagonal(qv_0.qs.get_unitary()))) < 1e-3)
True
