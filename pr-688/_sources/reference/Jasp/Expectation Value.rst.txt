.. _expectation_value:

Expectation Value
=================

.. currentmodule:: qrisp.jasp
.. autofunction:: expectation_value


Expectation Value for Hamiltonians
==================================

Estimating the expectation value of a Hamiltonian for a state is enabled by the respective ``.expectation_value`` methods for 
:meth:`QubitOperators <qrisp.operators.qubit.QubitOperator.expectation_value>` and :meth:`FermionicOperators <qrisp.operators.fermionic.FermionicOperator.expectation_value>`.

For example, we prepare the state

.. math::

    \ket{\psi_{\theta}} = (\cos(\theta)\ket{0}+\sin(\theta)\ket{1})^{\otimes 2}

::
            
    from qrisp import *
    from qrisp.operators import X,Y,Z
    import numpy as np

    def state_prep(theta):
        qv = QuantumFloat(2)

        ry(theta,qv)
    
        return qv

And compute the expectation value of the Hamiltonion $H=Z_0Z_1$ for the state $\ket{\psi_{\theta}}$

::

    @jaspify(terminal_sampling=True)
    def main():
            
        H = Z(0)*Z(1)

        ev_function = H.expectation_value(state_prep, precision = 0.01)

        return ev_function(np.pi/2)

    print(main())
    # Yields: 0.010126265783222899


