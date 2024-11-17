.. _QIRO:

.. currentmodule:: qrisp.qiro

Quantum Informed Recursive Optimization
=======================================

.. toctree::
    :hidden:
    
    QIROProblem
    QIROImplementations


An algorithm to facilitate the functionality of Quantum Informed Recursive Optimizations, as developed by J. Finzgar et. al. in `Quantum-Informed Recursive Optimization Algorithms (2023) <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.020327>`_ .

It is based on updating the problem instance based on correlations, that are in turn established with a QAOA protocol. For further info have a look at our :ref:`tutorial on QIRO!  <qiro_tutorial>`

The central data structure of the QIRO module is the :ref:`QIROProblem` class.

:ref:`QIROProblem`
------------------

The :ref:`QIROProblem` encapsulates the required prerequesites to run the algorithm:

* The ``problem`` to be solved, which is not necessarly a graph.
* The ``replacement_routine``, which has the job of performing the aforementioned specific reductions to the ``problem`` object.
* The ``cost_operator``, ``mixer``, ``init_function`` and ``cl_cost_function`` in analogy to :ref:`QAOAProblem` instantiation. 


.. _QIROMiXers:

Collection of mixers and auxiliary functions 
--------------------------------------------

.. autosummary::

   qiro_RXMixer
   qiro_init_function
   find_max
   

QIRO implementations of problem instances
-----------------------------------------

For implemented problem instances see :ref:`the QIRO implementations page <QIROImplementations>`





