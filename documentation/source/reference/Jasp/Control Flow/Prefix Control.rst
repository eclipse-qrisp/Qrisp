.. _prefix_control:

Prefix Control
==============

The following functions expose program control features from Jax as Jasp-compatible functions. While :ref:`jrange` and :ref:`ClControlEnvironment` work fine for many situations, their syntactically convenient form prohibit some cases that might be relevant to your application. In particular it is impossible to use "carry values", (ie. classical values that get computed during a loop or conditional) outside of the control structure. To realize this behavior you might be tempted to use the Jax-exposed functions `fori_loop <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html#jax.lax.fori_loop>`_, `while_loop <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html#jax.lax.while_loop>`_ and `cond <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.cond.html#jax.lax.cond>`_. However these do not properly track the Jasp internal quantum state, which is why we expose Jasp-compatible implementations of these functions.

.. note::

    If you `kernelized <quantum_kernel>`_ your code, the quantum state doesn't need to be tracked, implying you can use the Jax versions.

.. _q_fori_loop:

.. currentmodule:: qrisp.jasp
.. autofunction:: q_fori_loop

.. _q_while_loop:

.. currentmodule:: qrisp.jasp
.. autofunction:: q_while_loop

.. _q_cond_loop:

.. currentmodule:: qrisp.jasp
.. autofunction:: q_cond
