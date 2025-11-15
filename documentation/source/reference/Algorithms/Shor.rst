.. _Shor:

Shor's Algorithm
================

In the realm of quantum computing, where classical limitations are challenged and new horizons are explored, Shor's Algorithm stands as a testament to the transformative potential of quantum mechanics in the field of cryptography. Developed by mathematician Peter Shor in 1994, this groundbreaking algorithm has the power to revolutionize the world of cryptography by efficiently factoring large numbers—once considered an insurmountable task for classical computers.

Qrisp provides a dead simple interface to integer factorization using your own backend. For details how this algorithm is implemented, please check the `Shor's algorithm tutorial <https://www.qrisp.eu/general/tutorial/Shor.html>`_.

.. currentmodule:: qrisp.shor

.. autofunction:: shors_alg

.. _crypto_tools:

Cryptography tools
==================

These tools can be utilized to :ref:`spy on your enemies <ShorExample>`.

.. currentmodule:: qrisp.shor

.. autosummary::
   :toctree: generated/
   
   rsa_encrypt
   rsa_decrypt
   rsa_encrypt_string
   rsa_decrypt_string