.. _BlockEncoding:

BlockEncoding
=============

.. currentmodule:: qrisp.block_encodings
.. autoclass:: BlockEncoding

Creation
========

.. autosummary::
   :toctree: generated/

   BlockEncoding.from_operator

Utilities
=========

.. autosummary::
   :toctree: generated/
   
   BlockEncoding.apply
   BlockEncoding.apply_rus
   BlockEncoding.create_ancillas

Arithmetic
==========

.. autosummary::
   :toctree: generated/
   
   BlockEncoding.__add__
   BlockEncoding.__matmul__
   BlockEncoding.__mul__
   BlockEncoding.__neg__
   BlockEncoding.__sub__

Transformations
===============

.. autosummary::
   :toctree: generated/
   
   BlockEncoding.dagger
   BlockEncoding.qubitization
