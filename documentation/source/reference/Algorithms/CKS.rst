.. _CKS:

.. currentmodule:: qrisp.cks

Childs-Kothari-Somma (CKS)
==========================

The Childs-Kothari-Somma (CKS) algorithm is a Chebyshev polynomial- LCU-based approach for solving the Quantum Linear Systems Problem (QLSP), 
$A \vec{x} = \vec{b},$
which achieves an exponentially improved precision scaling compared to the original HHL algorithm.

.. math::

    A \vec{x} = \vec{b},

which significantly improves the precision scaling compared to the original HHL algorithm.

The goal is to prepare the quantum state

.. math::

    |\tilde{x}\rangle = A^{-1}\vec{b},

the solution to the linear system.

Unlike HHL, which depends polynomially on the precision parameter $\epsilon$ as $\mathcal{O}(1/\epsilon)$, the CKS algorithm achieves a polylogarithmic precision dependence by approximating the inverse function $\frac{1}{x}$ with a truncated Chebyshev series.

The computational complexity depends on the precision $\epsilon$, sparsity $d$ of the matrix $A$, and its condition number $\kappa$.

Overview of the CKS Algorithm
-----------------------------

The inverse operator $A^{-1}$ is approximated by a function $g(x)$ over the spectral range of $A$'s eigenvalues, expressed as a linear combination of odd Chebyshev polynomials of the first kind $A^{-1}\approx\sum_{k=1}^{2j_0+1}T_k(x)$:

.. math::

    g(x) = 4 \sum_{j=0}^{j_0} (-1)^j \alpha_j T_{2j+1}(x),

where \(j_0\) is the truncation order and \(\alpha_j > 0\) are positive coefficients.

The quantum implementation applies the Linear Combination of Unitaries (LCU) framework via the unitary:

.. math::

    W = V^\dagger U V,

where:

- $U$ applies the conditional Chebyshev polynomial operators $T_{2j+1}(A)$ block encoded through Qubitization.
- $V$ prepares the unary auxiliary register in a superposition weighted by the Chebyshec coefficients $\sqrt{\alpha_k}$.

.. autofunction:: CKS

Circuit construction and wrapper function
-----------------------------------------

.. autosummary::
   :toctree: generated/

   inner_CKS
   inner_CKS_wrapper

Miscellaneous
-------------

.. autosummary::
   :toctree: generated/

   CKS_parameters
   cheb_coefficients
   unary_angles
   unary_prep