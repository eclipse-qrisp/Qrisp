.. _BlockEncoding:

BlockEncoding
=============

.. currentmodule:: qrisp.block_encodings
.. autoclass:: BlockEncoding

Construction
------------

.. autosummary::
   :toctree: generated/

   BlockEncoding.from_array
   BlockEncoding.from_lcu
   BlockEncoding.from_operator

Utilities
---------

.. autosummary::
   :toctree: generated/
   
   BlockEncoding.apply
   BlockEncoding.apply_rus
   BlockEncoding.chebyshev
   BlockEncoding.create_ancillas
   BlockEncoding.dagger
   BlockEncoding.qubitization
   BlockEncoding.resources

Arithmetic
----------

.. autosummary::
   :toctree: generated/
   
   BlockEncoding.__add__
   BlockEncoding.__matmul__
   BlockEncoding.__mul__
   BlockEncoding.__neg__
   BlockEncoding.__sub__
   BlockEncoding.kron

Algorithms & Applications
-------------------------

.. autosummary::
   :toctree: generated/
   
   BlockEncoding.inv
   BlockEncoding.poly
   BlockEncoding.sim