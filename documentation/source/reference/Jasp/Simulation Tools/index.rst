.. _jasp_simulators:

.. note::
    
    This module is still under heavy development and the interface can therefore change at any time!

Simulation Tools
================

Jasp is tightly integrated within the Jax infrastructure, implying a variety of compilation and simulation techniques can be targeted.

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Simulator
     - Description
   * - :ref:`Jaspify <jaspify>`
     - A fast general purpose simulator that can use state sparsity. Doesn't perform well for demanding classical tasks.
   * - :ref:`QJIT <qjit>`
     - Calls the `Catalyst pipeline <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__ to compile and run the program via the Lightning simulator. Requires Catalyst to be installed.     
   * - :ref:`Boolean Simulation <boolean_simulation>`
     - Leverages the Jax pipeline to compile the program into a series of boolean operations. Restricted to programs that require only X, CX, CCX, etc. gates. This simulator can be extremely powerful for verifying the correctness of large classical subroutines, that are called by another quantum function.
   * - :ref:`Terminal Sampling <terminal_sampling>`
     - Uses the terminal sampling technique to perform accelerated quantum sampling. Some restrictions apply.

We encourage you to explore these simulators, as each one has their own perks/drawbacks. If your development is slowed down by slow simulation, feel free to try out a different simulation technique!

.. toctree::
   :maxdepth: 2
   :hidden:
   
   Jaspify
   QJIT
   Boolean Simulation
   Terminal Sampling
