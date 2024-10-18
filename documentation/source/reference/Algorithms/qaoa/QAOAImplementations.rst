.. _QAOAImplementations:

.. currentmodule:: qrisp.qaoa

Implementations
===============

.. toctree::
     :hidden:
    
     implementations/maxCut
     implementations/maxSat
     implementations/eThrLinTwo
     implementations/QUBO
     implementations/maxIndepSet
     implementations/maxClique
     implementations/maxSetPack
     implementations/minSetCover
     implementations/maxKColorableSubgraph


Our voyage into :ref:`MaxCut <MaxCutQAOA>` and :ref:`Max-$\\kappa$-Colorable Subgraph <MkCSQAOA>` problems is detailed in the :ref:`tutorial <tutorial>` section. 
The QAOA module is built with a focus on modularity, ensuring it can adapt to various problem instances while maintaining independence from the choice of decoding. 
This design approach makes it straightforwad to formulate and solve other problem instances taking the steps as we did in the tutorial. 

Here, you can find condensed presentations of various QAOA implementations:


.. list-table::
   :widths: 45 45 10
   :header-rows: 1

   * - PROBLEM INSTANCE
     - MIXER TYPE
     - IMPLEMENTED IN QRISP
   * - :ref:`MaxCut <maxCutQAOAdoc>`
     - X mixer
     -    ✅
   * - :ref:`Max-$\\ell$-SAT <maxsatQAOA>`
     - X mixer
     -    ✅
   * - :ref:`E3Lin2 <eThrLinTwoQAOA>`
     - X mixer
     -    ✅
   * - :ref:`QUBO <QUBOQAOAdoc>`
     - X mixer
     -    ✅ 
   * - :ref:`MaxIndependentSet <maxIndepSetQAOA>`
     - Controlled X mixer
     -    ✅
   * - :ref:`MaxClique <maxCliqueQAOA>`
     - Controlled X mixer
     -    ✅
   * - :ref:`MaxSetPacking <maxSetPackingQAOA>`
     - Controlled X mixer
     -    ✅
   * - :ref:`MinSetCover <minSetCoverQAOA>`
     - Controlled X mixer
     -    ✅
   * - :ref:`Max-$\\kappa$-Colorable Subgraph <MkCSQAOAdoc>`
     - XY mixer
     -    ✅ 





