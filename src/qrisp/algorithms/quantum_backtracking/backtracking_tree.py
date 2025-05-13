"""
\********************************************************************************
* Copyright (c) 2025 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************/
"""

from itertools import product

import numpy as np
import networkx as nx
from sympy.physics.quantum import Ket, OrthogonalKet

from qrisp.alg_primitives import QFT
from qrisp import (QuantumFloat, QuantumBool, QuantumArray, mcz, cx, h, ry, swap,
                    auto_uncompute, invert, control, IterationEnvironment, bin_rep,
                    multi_measurement, xxyy, p, QuantumVariable, cz,
                    mcx, z, x, RYGate, HGate, s, t, s_dg, t_dg)

"""
As specified in the paper (https://arxiv.org/abs/1509.02374), the key challenge
in implementing the quantum backtracking algorithm is the realization of the operators
R_A and R_B. These operators consists of the direct sum of diffuser operators D_x,
where x is an arbitrary node of the backtracking graph.
R_A and R_B are defined as the direct sum of these

R_A = [direct sum] D_x  [summed over all nodes x with even depth]
R_B = |r><r| [direct sum] D_x [summed over all nodes x with even depth]

Or in words: Each node x together with it's children {y : x->y}
defines a subspace H_x = span(|x>, {|y>, x->y}) on which the operator D_x is acting
in-place.

The definition of the D_x operators consists of multiple conditions:

    1. If x is an "accept" node, D_x = 1 (ie. the identity)
    2. If x is the root D_x = 1 - 2|psi_r><psi_r|
        Where |psi_r> = (1+d_r*n)**-0.5 * (|r> + n**0.5 * [sum] |y>)
        Where d_r is  the degree of the root, n is the maximum depth of the tree
        and the sum iterates over all children of r
    3. Otherwise: D_x = 1 - 2 |psi_x><psi_x|
        Where |psi_x> = (d_x)**-0.5 (|x> + [sum] |y>)
        Where d_x is the degree of x and the sum iterates over all children of x

To implement this operator, we will rewrite it a bit:

D_x = 1 - (1+(-1)**accept(x)) * |psi_r><psi_r|

The next step is to assume an operator U_x, which prepares |psi_x> from x

|psi_x> = U_x |x>

We can then write

D_x = D_x = 1 - (1+(-1)**accept(x)) * |psi_r><psi_r|
    = U_x (1 - (1+(-1)**accept(x))*|x><x|) U_x^(-1)

If we pick an encoding, where each node state |x> is a computational basis state,
the center bracket (1 - (1+(-1)**accept(x))*|x><x|) can be realized as a
Z-gate on the bitstring on the
result of accept(x).

Furthermore we need to make sure that only the phase of |x> is flipped. If we simply
apply a Z-gate onto the result of accept, we could also flip the phase of a 
child node y with accept(y) = True. We remedy this problem by determining wether
|x> is odd into a qubit called oddity. For the even diffusing function, this qubit
needs to be in the False state, for the odd function this qubit needs to be in the
True state.

This qubit is then used in a CZ gate.

D_x = U_x CZ(accept(x), oddity(x)) U_x^(-1)

The operator U_x is implemented as the function psi_prep below.

The operator D_x is implemented as the method qstep_diffuser of the
QuantumBacktrackingTree class.

The next layer of complexity is the reject function.
If a node is rejected, it has no children, ie. |psi_x> = |x>.

This could be realized by modifiying the operator U_x but as it turns out there is
a more efficient possibility.

For this consider the fact that for a rejected node

D_x = 1 - 2 |psi_x><psi_x| = 1 - 2 |x><x|

The algorithm will however never evaluate this operator on any of the children of x
so restricted on H_x we can assume

D_x = -1

We therfore now need to modify D_x such that D_x = -1 when reject(x) is True.

To achieve this we deploy a circuit that integrates with our previous construction
but instead performs 

D_x = U_x (-1) U_x^(-1)

if reject(x) = True.

Our goal is therefore now to find some operator O such that

CZ(accept(x), oddity(x)) O(x) = [-1 if reject(x) is True] 
                                        [CZ(accept(x), oddity(x)) if reject(x) is False]
                                        
We conclude that O(reject(x)) needs to flip the phases of all the (pseudo) child states.
(because H_x = span(|x>, {|y>, x->y})).

Similar to the above approach we reuse the oddity qubit to make sure we only operate
on the child states.

Unfortunately we can't assume that the reject function also returns True on the children,
therefore we "lift" the state. This means that we evaluate the reject function on the 
parent. We denote this with the symbol ^.

We conclude:
    
O(x) = CZ(reject(x^), not oddity(x))

To understand how the lifting works, please check the implementation of qstep_diffuser.


This brings us to another important point in the implementation of the algorithm:
The encoding of the node states.

In principle, the paper makes no statement how such an encoding could be realized.
The approach we took here is the following:

A node state |x> is specified by:
    1. A one-hot encoded integer h (for height), which specifies the distance of 
        x from the the steepest leaf.
    2. A QuantumArray branch_qa, which specifies the path to take to reach x.

A few things have to be said about this encoding:

We choose to encoded the height instead of the distance of the root 
(as described in the paper), as this allows straight forward generation of subtrees. 
A node which has height 4 in a some tree T, still has height 4 in a subtree of T.

The initial state of branch_qa is |0>|0>|0>...

Because we use the height instead of the path length, the path to the node is 
specified by the reveresed array. Ie. the path [1,1,0,1]  in a depth 7 tree is 
specified by the branch_qa state

|0>|0>|0>|1>|0>|1>|1>

Any state where there is a non-zero state at an index lower than h is considered
"non-algorithmic", ie. the state does not represent a node. An example could be

h = |4>
branch_qa = |0>|0>|1>|0>|0>|0>
"""


class QuantumBacktrackingTree:
    r"""
    This class describes the central data structure to run backtracking algorithms in
    a quantum setting. `Backtracking algorithms <https://en.wikipedia.org/wiki/Backtracking>`_
    are a very general class of algorithms which cover many problems of combinatorial
    optimization such as 3-SAT or TSP.

    Backtracking algorithms can be put into a very general form. Given is a maximum
    recursion depth, two functions called ``accept``/``reject`` and the set of
    possible assignments for an iterable x.

    ::

        from problem import accept, reject, max_depth, assignments

        def backtracking(x):

            if accept(x):
                return x

            if reject(x) or len(x) == max_depth:
                return None

            for j in assigments:
                y = list(x)
                y.append(j)
                res = backtracking(y)
                if res is not None:
                    return res


    The power of these algorithms lies in the fact that they can quickly discard
    large parts of the potential solution space by using the reject function to
    cancel the recursion. Compared to an unstructured search, where only the
    accept function is available, this can significantly cut the required resources.

    The quantum algorithm for solving these problems has been
    `proposed by Ashley Montanaro <https://arxiv.org/abs/1509.02374>`_ and yields
    a 1 to 1 correspondence between an arbitrary classical backtracking algorithm
    and it's quantum equivalent. The quantum version achieves a quadratic speed up
    over the classical one.

    The algorithm is based on performing quantum phase estimation on a quantum walk
    operator, which traverses the backtracking tree. The core algorithm returns
    "Node exists" if the 0 component of the quantum phase estimation result
    has a higher probability then 3/8 = 0.375.

    Similar to the classical version, for the Qrisp implementation of this quantum
    algorithm, a backtracking problem is specified by a maximum recursion depth
    and two functions, each returning a :ref:`QuantumBool` respectively:

    **accept**: Is the function that returns True, if called on a node, satisfying the
    specifications.

    **reject**: Is the function that returns True, if called on a node, representing a
    branch that should no longer be considered.

    Furthermore required is a :ref:`QuantumVariable` that specifies the branches
    that can be taken by the algorithm at each node.


    **Node encoding**
    
    An important aspect of this algorithm is the node encoding. In Montanaros
    paper a central quantity is the distance from the root $l(x)$. This however
    doesn't generalize well to the specification of subtrees, which is why
    we encode the height of a node. For example in a tree with maximum depth $n$
    a leaf has height 0 and the root has height $n$.
    
    This quantity is encoded as a one-hot integer QuantumVariable, which can be
    found under the attribute ``h``.
    
    To fully identify a node, we also need to specify the path to take starting
    at the root. This path is encoded in a :ref:`QuantumArray`, which can be found
    under the attribute ``branch_qa``. To fit into the setting of height encoding,
    this array contains the reversed path.
    
    We summarize the encoding by giving an example:
        
    In a binary tree with depth 5, the node that has the path from the root [1,1]
    is encoded by

    .. math::
        
        \begin{align}
        \ket{\text{branch_qa}} &= \ket{0}\ket{0}\ket{0}\ket{1}\ket{1}\\
        \ket{\text{h}} &= \ket{3} = \ket{00010}\\
        \ket{x} &= \ket{\text{branch_qa}}\ket{\text{h}}
        \end{align}
    

    **Details on the predicate functions**
    
    The predicate functions ``accept`` and ``reject`` must meet certain conditions
    for the algorithm to function properly:
        
    * Both functions have to return a :ref:`QuantumBool`.
    * Both functions must not change the state of the tree.
    * Both functions must delete/uncompute all temporarily created QuantumVariables.
    * ``accept`` and ``reject`` must never return ``True`` on the same node.
    
    The ``subspace_optimization`` keyword enables a significant optimization of
    the ``quantum_step`` function. This keyword can be set to True if the ``reject``
    function is guaranteed to return the value ``reject(x)`` also on the non-algorithmic
    subspace of x. For instance, if x = [0,0,1] in a depth 4 tree, the encoded state is
    
    .. math::
        
        \begin{align}
        \ket{\text{branch_qa}} &= \ket{0}\ket{1}\ket{0}\ket{0}\\
        \ket{\text{h}} &= \ket{1}\\
        \ket{x} &= \ket{\text{branch_qa}}\ket{\text{h}}
        \end{align}
    
    A state from the non-algorithmic subspace of x is now a state that has non-zero
    entries in ``branch_qa`` at indices less than ``h`` ie.
    
    .. math::
        
        \begin{align}
        \ket{\text{branch_qa}_{NA}} &= \ket{1}\ket{1}\ket{0}\ket{0}\\
        \ket{\text{h}} &= \ket{1}\\
        \ket{\tilde{x}} &= \ket{\text{branch_qa}_{NA}}\ket{\text{h}}
        \end{align}
        
    For the ``subspace_optimization`` to return proper results, the ``reject``
    function must therefore satisfy:
        
    .. math::
        
        \text{reject}(\ket{x}) = \text{reject}(\ket{\tilde{x}})



    .. note::

        Many implementations of backtracking also include the possibility for
        deciding which entries of x to assign based on some user provided heuristic.
        The quantum version also supports this feature, however it is not yet
        implemented in Qrisp.

    Parameters
    ----------

    max_depth : integer
        The depth of the backtracking tree.
    branch_qv : QuantumVariable
        A QuantumVariable representing the possible branches of each node.
    accept : function
        A function taking an instance of QuantumBacktrackingTree and returning
        a QuantumBool, which is ``True``, if called on a satisfying node.
    reject : function
        A function taking an instance of QuantumBacktrackingTree and returning
        a QuantumBool, which is ``True``, if a called on a node where the corresponding
        branch should no longer be investigated.
    subspace_optimization : bool, optional
        If set to ``True``, a significant optimization of the ``quantum_step`` function 
        will be applied. The reject function has to fullfil a certain property
        for this to yield the correct results. Please check the "Details on the
        predicate functions" section for more information. The default is ``False``.


    Attributes
    ----------

    h : :ref:`QuantumFloat`
        A one hot encoded integer representing the height of the node. The root
        has ``h = max_depth``, it's children have ``h = max_depth-1`` etc.
    branch_qa : :ref:`QuantumArray`
        A QuantumArray representing the path from the root to the current node.
        The qtype of this QuantumArray is what is been provided as ``branch_qv``.
        Note that the state of this array is the reversed path, ie. a the node
        with path ``[1,1,0,1]`` in a depth 7 tree has the state:
        $\ket{0}\ket{0}\ket{0}\ket{1}\ket{0}\ket{1}\ket{1}$
        States that have a non-zero value at entries indexed smaller than ``h``,
        are considered non-algorithmic and will never be visited
        (eg. h=3, branch_qa = $\ket{1}\ket{1}\ket{1}\ket{1}$).
    qs : :ref:`QuantumSession`
        The QuantumSession of the backtracking tree.
    max_depth : int
        An integer specifying the maximum depth of each node.


    Examples
    --------

    **Checking for the existence of a solution**

    Even though primary purpose of backtracking algorithms is to find a solution,
    at the core, Montanaros algorithm only determines solution existence. This can
    however still be leveraged into a solution finding algorithm.

    To demonstrate the solution existence functionality, we search the binary
    tree that consists only of nodes with alternating branching.
    We accept if we find the node ``[0,0,1]`` (doesn't exist in this tree).

    For this we first set up the reject condition.

    ::

        from qrisp import *

        @auto_uncompute
        def reject(tree):
            
            oddity = QuantumBool()
            for i in range(tree.h.size):
                if i%2:
                    cx(tree.h[i], oddity)
            
            parity = QuantumBool()
            
            for i in range(tree.branch_qa.size):
                cx(tree.branch_qa[i], parity)
            
            exclude_init = (tree.h < tree.max_depth-1)
                
            return exclude_init & (oddity != parity)

    This function determines first determines the oddity of the height parameter
    (remember ``tree.h`` has one-hot encoding!). Next the parity of the branching 
    path is evaluated. Parity means "is the amount of ones in the path even or odd".
    
    We will reject the node if the oddity of is unequal to the parity and therefore
    reject any path that took a 1 after it already took a 1 (same for 0).
    
    On the root and it's children there will be rejection to allow for two different
    paths.

    We now implement the accept condition:

    ::

        @auto_uncompute
        def accept(tree):
            
            height_condition = (tree.h == 0)
            
            path_condition = QuantumBool()
            mcx(tree.branch_qa[::-1], path_condition, ctrl_state = "001")
        
            return height_condition & path_condition


    Subsequently we set up the class instance:

    ::

        from qrisp.quantum_backtracking import QuantumBacktrackingTree

        tree = QuantumBacktrackingTree(max_depth = 3,
                                       branch_qv = QuantumFloat(1),
                                       accept = accept,
                                       reject = reject)

        tree.init_node([])

    We can evaluate the statevector:

    >>> tree.statevector()
    1.0*|[]>

    The ``[]`` indicates that this is the root state. If the tree was in the state
    of a child of the root (say the one connected to the 1 branch) it would be ``[1]``.

    Note that the ``statevector`` method decodes the QuantumVariables holding the
    node state for convenient readibility. If you want to see the encoded variables
    you can take a look at the :ref:`QuantumSession` s :meth:`statevector method<qrisp.QuantumSession.statevector>`:

    >>> tree.qs.statevector()
    |0>**3*|3>

    We can also visualize the statevector of the tree:

    >>> import matplotlib.pyplot as plt
    >>> tree.visualize_statevector()
    >>> plt.show()

    .. image:: ./root_state_plot.png
        :width: 200
        :alt: Root statevector plot
        :align: left

    |
    |
    |
    |
    |

    And finally evaluate the algorithm:

    ::

        qpe_res = tree.estimate_phase(precision = 4)

    Perform a measurement

    >>> mes_res = qpe_res.get_measurement()
    >>> mes_res[0]
    0.1036

    The 0 component has only 10.36% probability of appearing, therefore we can conclude,
    that in the specified tree no such node exists.

    We now perform the same process but with a trivial reject function:

    ::

        def reject(tree):
            return QuantumBool()

        tree = QuantumBacktrackingTree(max_depth = 3,
                                       branch_qv = QuantumFloat(1),
                                       accept = accept,
                                       reject = reject)

        tree.init_node([])

        qpe_res = tree.estimate_phase(precision = 4)


    >>> mes_res = qpe_res.get_measurement()
    >>> mes_res[0]
    0.5039

    We see a probability of more than 50%, implying a solution exists in
    this tree.

    **Finding a solution**

    Montanaros approach to determine a solution is to classically traverse the tree,
    by always picking the child node where the quantum algorithm returns "Node exists".
    Finding a solution can therefore be considered a hybrid algorithm.

    To demonstrate, we search for the node ``[1,1,1]`` with a trivial reject function.

    ::

        @auto_uncompute
        def accept(tree):
            height_condition = (tree.h == tree.max_depth - 3)
            path_condition = QuantumBool()
            mcx(tree.branch_qa[-3:], path_condition)

            return height_condition & path_condition

        def reject(tree):
            return QuantumBool()


    Set up the QuantumBacktrackingTree instance:


    >>> max_depth = 4
    >>> tree = QuantumBacktrackingTree(max_depth,
                                       branch_qv = QuantumFloat(1),
                                       accept = accept,
                                       reject = reject)

    And call the solution finding algorithm:

    >>> tree.find_solution(precision = 5)
    [1, 1, 1]
    
    **Using the subspace_optimization keyword**
    
    To demonstrate the usage of this feature, we create two tree instances - one
    with and one without the optimization.
    
    ::
        
        def accept(tree):
            return QuantumBool()
        
        def reject(tree):
            return QuantumBool()

    >>> opt_tree = QuantumBacktrackingTree(3, branch_qv = QuantumFloat(1), accept = accept, reject = reject, subspace_optimization = True)
    >>> no_opt_tree = QuantumBacktrackingTree(3, branch_qv = QuantumFloat(1), accept = accept, reject = reject, subspace_optimization = False)

    Perform a ``quantum_step`` on both of them:
        
    >>> opt_tree.quantum_step()
    >>> no_opt_tree.quantum_step()
    
    And evaluate some benchmarks:
    
    >>> no_opt_tree.qs.compile().depth()
    89
    >>> no_opt_tree.qs.compile().cnot_count()
    68
    
    With the optimization these values are much better:
        
    >>> opt_tree.qs.compile().depth()
    48
    >>> opt_tree.qs.compile().cnot_count()
    38
            
    """

    def __init__(self, max_depth, branch_qv, accept, reject, subspace_optimization = False):

        self.max_depth = max_depth

        self.degree = 2**branch_qv.size

        self.branch_qa = QuantumArray(qtype=branch_qv, shape=max_depth)

        self.h = OHQInt(max_depth+1, name="h*", qs=self.branch_qa.qs)

        self.qs = self.h.qs

        self.accept_function = accept
        self.reject_function = reject
        
        self.subspace_optimization = subspace_optimization

    def accept(self):
        return self.accept_function(self)

    def reject(self):
        return self.reject_function(self)

    @auto_uncompute
    def qstep_diffuser(self, even, ctrl=[], min_height_assumption = 0):
        """
        Performs the operators :math:`R_A` or :math:`R_B`. For more information on these operators check `the paper <https://arxiv.org/abs/1509.02374>`_.

        Parameters
        ----------
        even : bool
            Depending on the parameter, the diffuser acts on the subspaces $\mathcal H_x=\{\ket{x}\}\cup\{\ket{y}\,|\,x\\rightarrow y\}$ where $x$ has odd (``even=False``) or even (``even=True``) height.
            Note that "even" refers to the parity of the ``h`` attribute instead of the distance from the root.
            If the ``max_depth`` of the tree is odd, and ``even=False`` then $R_A$ (otherwise $R_B$) is performed, and vice verse if the ``max_depth`` is even. 

        ctrl : List[Qubit], optional
            A list of qubits that allows performant controlling. The default is [].

        Examples
        --------

        We set up a QuantumBackTrackingTree and perform the diffuser on a marked node

        ::

            from qrisp import auto_uncompute, QuantumBool, QuantumFloat
            from qrisp.quantum_backtracking import QuantumBacktrackingTree

            @auto_uncompute
            def reject(tree):
                return QuantumBool()

            @auto_uncompute
            def accept(tree):
                return (tree.h == 1)


            tree = QuantumBacktrackingTree(3, QuantumFloat(
                1, name = "branch_qf*"), accept, reject)

            tree.init_node([1,1])

        >>> print(tree.qs.statevector())
        |0>*|1>**3
        >>> tree.qstep_diffuser(even = False)
        >>> print(tree.qs.statevector())
        |0>*|1>**3

        We see that the node (as expected) is invariant under :math:`R_A`.
        """

        # This function performs the operation
        # D_x = U_x (1 - (1+(-1)**accept(x))*|x><x|) U_x^(-1)
        # For more information, check the beginning of this file

        # Perform U_x^(-1)
        with invert():
            psi_prep(self, even=even, min_height_assumption = min_height_assumption)

        # We now perform the operation
        # 1 - (1+(-1)**accept(x))*|x><x|
        # by executing an appropriate mcz gate

        # An additional detail to consider:
        #   We allow for additional control qubits. This is important because
        #   this operator will undergo phase estimation and we want to prevent
        #   performing automatic synthesis of the controlled operation,
        #   as this would imply controlling EVERY gate. Instead we can just
        #   control the mcz gate.

        
        
        # Prepare control state specificator
        ctrl_state = ""
        mcz_list = []

        # D_x operates on the space span(|x>, {|y>, x->y})
        # In order to make sure our mcz gate only marks |x>, we can use the
        # oddity of h, because if h(x) is odd, then h(y) is not.

        oddity_qbl = QuantumBool()
        
        for i in range(self.max_depth+1):
            if i < min_height_assumption:
                continue
            if bool(i%2) != even:
                cx(self.h[i], oddity_qbl)
                

        ctrl_state += "1"
        mcz_list.append(oddity_qbl)

        # Determine accept value
        accept_value = self.accept()
        mcz_list.append(accept_value)
        ctrl_state += "0"


    	# Add additional control qubits
        mcz_list += ctrl
        ctrl_state += "1"*len(ctrl)
        
        # Perform mcz gate
        mcz(mcz_list, ctrl_state=ctrl_state)
        
        
        # We now perform the phase-flip on the child states.
        # For more details why this yields the correct behavior please consult
        # the text at the beginning of this file.
        
        # For this we first have to perform the lifting operation.
        # Lifting means, that the child states |y> are mapped to their parent.
        
        # The first step to achieve this is to swap the branch information
        # into a temporary container. This way the branching information is 0.

        # Check if |x> is root.
        
        
        # This
        if self.max_depth%2 == even:
            cx(self.h[self.max_depth],oddity_qbl)
        
        # Instead of this
        
        # is_root = QuantumBool()
        # cx(self.h[self.max_depth],is_root)
        
        temporary_container = self.branch_qa.qtype.duplicate()

        for i in range(self.max_depth):
            if i < min_height_assumption or self.subspace_optimization:
                continue
            if bool(i%2) == even:
                with control(self.h[i], ctrl_method = "gray_pt"):
                    swap(temporary_container, self.branch_qa[i])
        
        # The second step is to increment h. Due to the one-hot encoding of h,
        # we can do this for free with a compiler swap.
        self.h.reg.insert(0, self.h.reg.pop(-1))
        
        # Determine reject value
        reject_value = self.reject_function(self)
        mcz_list = [reject_value]
        ctrl_state = "1"

        # Make sure we only apply the phase to the child states (parent states
        # have oddity 1)
        mcz_list.append(oddity_qbl)
        ctrl_state += "0"

        # Check if |x> is root. Otherwise, if the reject funtions returns "True" on the lift of the root a wrong phase (-1) may be applied to the root.
        # mcz_list.append(is_root)
        # ctrl_state += "0"
        

        # Add extra controls
        mcz_list += ctrl
        ctrl_state += "1"*len(ctrl)
        
        #Perform MCZ gate
        mcz(mcz_list, ctrl_state = ctrl_state)
        
        #Reverse compiler swap
        self.h.reg.append(self.h.reg.pop(0))

        #Reintroduce branching information
        for i in range(self.max_depth):
            if i < min_height_assumption or self.subspace_optimization:
                continue
            if bool(i%2) == even:
                with control(self.h[i], ctrl_method = "gray_pt_inv"):
                    swap(temporary_container, self.branch_qa[i])

        #Delete temporary container.
        temporary_container.delete()
        
        # Perform U_x
        psi_prep(self, even=even, min_height_assumption = min_height_assumption)

    def quantum_step(self, ctrl=[], min_height_assumption = 0):
        """
        Performs the quantum step operator $R_BR_A$.
        For more information check the :meth:`diffuser method <qrisp.quantum_backtracking.QuantumBacktrackingTree.qstep_diffuser>`.

        Parameters
        ----------
        ctrl : List[Qubit], optional
            A list of qubits, the step operator should be controlled on. The default is [].
        """

        self.qstep_diffuser(even=not self.max_depth % 2, ctrl=ctrl, min_height_assumption = min_height_assumption)
        self.qstep_diffuser(even=self.max_depth % 2, ctrl=ctrl, min_height_assumption = min_height_assumption - 1)

    def estimate_phase(self, precision):
        r"""
        Performs :meth:`quantum phase estimation <qrisp.QPE>` on the :meth:`quantum step operator <qrisp.quantum_backtracking.QuantumBacktrackingTree.quantum_step>`.

        If executed with sufficient precision, the phase estimation will yield a QuantumFloat, where the probability of the 0 component indicates the presence of a node where the ``accept`` function yielded ``True``.

        If the probability is higher than 3/8 :math:`\Rightarrow` A solution exists.

        If the probability is less than 1/4 :math:`\Rightarrow` No solution exists.

        Otherwise :math:`\Rightarrow` Increase precision.

        In general, the required precision is proportional to

        .. math::

            \frac{\text{log}_2(Tn)}{2} + \beta

        Where :math:`T` is the amount of nodes, that would be visited by a classical algorithm, :math:`n` is the maximum depth and :math:`\beta` is a universal constant.


        Parameters
        ----------
        precision : int
            The precision to perform the quantum phase estimation with.

        Returns
        -------
        qpe_res : :ref:`QuantumFloat`
            The QuantumFloat containing the result of the phase estimation.

        """

        qpe_res = QuantumFloat(precision, -precision, qs = self.qs)

        h(qpe_res)
        
        from qrisp import check_if_fresh
        
        if check_if_fresh(self.h.reg[:-1], self.qs):
            height_tracker = int(self.max_depth) + 1
        else:
            height_tracker = -1
            
        for i in range(qpe_res.size):
            
            if height_tracker >= 0 and False:
                for j in range(2**i):
                    self.quantum_step(ctrl=[qpe_res[i]], min_height_assumption = height_tracker)
                    height_tracker -= 2
            else:
                with IterationEnvironment(self.qs, 2**i, precompile=True):
                    self.quantum_step(ctrl=[qpe_res[i]])

        QFT(qpe_res, inv=True)

        return qpe_res


    def init_phi(self, path):
        r"""
        Initializes the normalized version of the state :math:`\ket{\phi}`.

        .. math::

            \ket{\phi} = \sqrt{n}\ket{r} + \sum_{x \neq r, \\ x \rightsquigarrow x_0} (-1)^{l(x)} \ket{x}

        Where :math:`x \rightsquigarrow x_0` means that :math:`x` is on the path from :math:`r` to :math:`x_0` (including :math:`x_0`).

        If :math:`x_0` is a marked node, this state is invariant under the quantum step operator.

        Parameters
        ----------
        path : List
            The list of branches specifying the path from the root to :math:`x_0`.

        Examples
        --------

        We set up a backtracking tree of depth 3, where the marked element is the 111 node.

        ::

            from qrisp import auto_uncompute, QuantumBool, QuantumFloat
            from qrisp.quantum_backtracking import QuantumBacktrackingTree

            @auto_uncompute
            def reject(tree):
                return QuantumBool()

            @auto_uncompute
            def accept(tree):
                return (tree.branch_qa[0] == 1) & (tree.branch_qa[1] == 1) & (tree.branch_qa[2] == 1)


            tree = QuantumBacktrackingTree(3, QuantumFloat(
                1, name = "branch_qf*"), accept, reject)

        Initialize :math:`\ket{\phi}` and evaluate the statevector:

        >>> tree.init_phi([1,1,1])
        >>> print(tree.qs.statevector())
        (0.816496014595032*|0>*|1>**3 - 0.816496014595032*|0>**2*|1>*|2> + 1.0*sqrt(2)*|0>**3*|3> - 0.816496014595032*|1>**3*|0>)/2

        Perform the quantum step and evaluate the statevector again:

        >>> tree.quantum_step()
        >>> print(tree.qs.statevector())
        (0.816496014595032*|0>*|1>**3 - 0.816496014595032*|0>**2*|1>*|2> + 1.0*sqrt(2)*|0>**3*|3> - 0.816496014595032*|1>**3*|0>)/2

        We see that the node (as expected) is invariant under the quantum step operator.
        """

        h_state = {}
        h_state[self.max_depth] = (self.max_depth)**0.5

        for i in range(1, len(path)+1):
            h_state[self.max_depth - i] = (-1)**(i)

        self.h[:] = h_state

        for i in range(1, len(path)+1):
            with self.h == self.max_depth - i:
                for j in range(i):
                    self.branch_qa[-j-1].encode(path[j], permit_dirtyness=True)

    def init_node(self, path):
        """
        Initializes the state of a given node.

        Parameters
        ----------
        path : List
            List of the branch labels indicating the path from the root to the node.

        Examples
        --------

        We initialize a backtracking tree in the 101 node.

        ::

            from qrisp import auto_uncompute, QuantumBool, QuantumFloat
            from qrisp.quantum_backtracking import QuantumBacktrackingTree

            @auto_uncompute
            def reject(tree):
                return QuantumBool()

            @auto_uncompute
            def accept(tree):
                return QuantumBool()

            tree = QuantumBacktrackingTree(3, QuantumFloat(
                1, name = "branch_qf*"), accept, reject)

            tree.init_node([1,0,1])

        """

        self.h[:] = self.max_depth - len(path)
        if len(path):
            self.branch_qa[-len(path):] = path[::-1]

    def subtree(self, new_root):
        """
        Returns the subtree of a given node.

        Parameters
        ----------
        new_root : list
            The path from the root of self to the root of the subtree.

        Returns
        -------
        QuantumBacktrackingTree
            The subtree starting at the specified root.

        Examples
        --------

        We initiate a QuantumBacktrackingTree with trivial reject
        function and create a subtree starting at an accepted node-

        ::

            from qrisp import auto_uncompute, QuantumBool, QuantumFloat
            from qrisp.quantum_backtracking import QuantumBacktrackingTree

            @auto_uncompute
            def accept(tree):
                height_cond = (tree.h == 2)
                return height_cond

            @auto_uncompute
            def reject(tree):
                return QuantumBool()


        Create and initiate the parent tree.

        >>> depth = 5
        >>> tree = QuantumBacktrackingTree(depth, QuantumFloat(1, name = "branch_qf*"), accept, reject)
        >>> tree.init_node([])
        >>> print(accept(tree))
        {False: 1.0}

        We now create the subtree, where the new root has height two, ie. the accept
        function returns ``True``.

        >>> subtree = tree.subtree([0,1,0])
        >>> subtree.init_node([])
        >>> print(accept(subtree))
        {True: 1.0}

        """
        return Subtree(self, new_root)

    def copy(self):
        """
        Returns a copy of self. Copy means a QuantumBacktrackingTree with identical
        depth, accept/reject functions etc. but with freshly allocated QuantumVariables.

        Returns
        -------
        QuantumBacktrackingTree
            Another instance with the same depth/accept/reject etc.

        """
        return Subtree(self, [])

    def find_solution(self, precision, cl_accept=None, measurement_kwargs={}):
        """
        Determines a path to a solution.

        Parameters
        ----------
        precision : integer
            The precision to perform the quantum phase estimation(s) with.
        cl_accept : function, optional
            A classical version of the accept function of self. Needs to
            receive a list to indicate a path and returns a bool wether the
            node is accepted. By default, the accept function of self will be
            evaluated on a simulator.
        measurement_kwargs : dictionary
            A dictionary to give keyword arguments that specify how measurements
            are evaluated. The default is {}.

        Returns
        -------
        List
            A list indicating the path to a node where the accept function
            returns True.

        Examples
        --------

        We create a accept function that marks the node [0,1] and a trivial
        reject function.

        ::

            from qrisp import auto_uncompute, QuantumBool, QuantumFloat, mcx
            from qrisp.quantum_backtracking import QuantumBacktrackingTree

            @auto_uncompute
            def accept(tree):
                height_cond = (tree.h == 1) # The [0,1] node has height 1
                path_cond = QuantumBool()
                mcx(list(tree.branch_qa)[1:], path_cond, ctrl_state="10")
                return path_cond & height_cond

            @auto_uncompute
            def reject(tree):
                return QuantumBool()


        Create backtracking tree object:

        >>> depth = 3
        >>> tree = QuantumBacktrackingTree(depth, QuantumFloat(1, name = "branch_qf*"), accept, reject)

        Find solution

        >>> res = tree.find_solution(4)
        >>> print(res)
        [0, 1]

        """

        return find_solution(self, precision, cl_accept, measurement_kwargs=measurement_kwargs)


    def path_decoder(self, h, branch_qa):
        """
        Returns the path representation for a given constellation of the
        ``h`` and ``branch_qa`` variables. The path representation is a list
        indicating which branches to take starting from the root.
        This function exists because the encoding of the nodes is hardware
        efficient but inconvenient for humans to read.

        Parameters
        ----------
        h : integer
            The integer describing the height of the node.
        branch_qa : list
            The list of branches to take to reach the root, starting from the node.

        Returns
        -------
        list
            The list of path variables to take to reach the node, starting
            from the root.

        Examples
        --------

        We create a QuantumBacktrackingTree, initiate a node and retrieve the path.

        ::

            from qrisp import auto_uncompute, QuantumBool, QuantumFloat, multi_measurement
            from qrisp.quantum_backtracking import QuantumBacktrackingTree

            @auto_uncompute
            def accept(tree):
                return QuantumBool()

            @auto_uncompute
            def reject(tree):
                return QuantumBool()

        >>> depth = 5
        >>> tree = QuantumBacktrackingTree(depth, QuantumFloat(1, name = "branch_qf*"), accept, reject)
        >>> tree.init_node([1,0])
        >>> multi_measurement([tree.h, tree.branch_qa])
        {(3, OutcomeArray([0, 0, 0, 0, 1])): 1.0}

        Retrieve the path

        >>> tree.path_decoder(3, [0, 0, 0, 0, 1])
        [1, 0]

        """


        l = self.max_depth - h
        return list(branch_qa[::-1][:l])




    def statevector_graph(self, return_root=False):
        r"""
        Returns a NetworkX Graph representing the quantum state of the backtracking tree.
        The nodes have an ``amplitude`` attribute, indicating the complex amplitude of that node.

        Parameters
        ----------
        return_root : bool, optional
            If set to ``True``, this method will also return the root node. The default is False.

        Returns
        -------
        networkx.DiGraph
            A graph representing the statevector.

        Examples
        --------

        We initialize a backtracking tree, initialize a :math:`\ket{\phi}` state and retrieve
        the statevector graph.

        ::

            from qrisp import auto_uncompute, QuantumBool, QuantumFloat
            from qrisp.quantum_backtracking import QuantumBacktrackingTree

            @auto_uncompute
            def reject(tree):
                return QuantumBool()

            @auto_uncompute
            def accept(tree):
                return (tree.branch_qa[0] == 1) & (tree.branch_qa[1] == 1) & (tree.branch_qa[2] == 1)


        Create backtracking tree and initialize :math:`\ket{\phi}`.

        >>> tree = QuantumBacktrackingTree(3, QuantumFloat(1, name = "branch_qf*"), accept, reject)
        >>> tree.init_phi([1,1,1])
        >>> statevector_graph = tree.statevector_graph()
        >>> print(statevector_graph.nodes())
        [QBTNode(path = [], amplitude = (0.7071066+0j)), QBTNode(path = [0], amplitude = 0j), QBTNode(path = [1], amplitude = (-0.408248-4.4703484e-08j)), QBTNode(path = [0, 0], amplitude = 0j), QBTNode(path = [0, 1], amplitude = 0j), QBTNode(path = [1, 0], amplitude = 0j), QBTNode(path = [1, 1], amplitude = (0.4082481+1.4901161e-08j)), QBTNode(path = [0, 0, 0], amplitude = 0j), QBTNode(path = [0, 0, 1], amplitude = 0j), QBTNode(path = [0, 1, 0], amplitude = 0j), QBTNode(path = [0, 1, 1], amplitude = 0j), QBTNode(path = [1, 0, 0], amplitude = 0j), QBTNode(path = [1, 0, 1], amplitude = 0j), QBTNode(path = [1, 1, 0], amplitude = 0j), QBTNode(path = [1, 1, 1], amplitude = (-0.40824816+5.9604645e-08j))]
        >>> statevector_graph, root = tree.statevector_graph(return_root = True)
        >>> print(root)
        QBTNode(path = [], amplitude = (0.7071066+0j))

        """

        from networkx import DiGraph

        sv_function = self.qs.statevector("function")

        res_graph = DiGraph()

        root = QBTNode(self, [])

        root.amplitude = sv_function(root.sv_specifier())

        res_graph.add_node(root)

        last_layer = [root]


        for i in range(self.max_depth):

            next_layer = []

            for parent_node in last_layer:

                for j in range(2**self.branch_qa[0].size):

                    child_node_path = list(parent_node.path) + [self.branch_qa[0].decoder(j)]

                    child_node = QBTNode(self, child_node_path)

                    child_node.amplitude = sv_function(child_node.sv_specifier())

                    res_graph.add_node(child_node)
                    res_graph.add_edge(parent_node, child_node,
                                       label=child_node_path[-1])

                    next_layer.append(child_node)

            last_layer = next_layer

        if return_root:
            return res_graph, root
        return res_graph

    def visualize_statevector(self, pos=None):
        """
        Visualizes the statevector graph.

        Parameters
        ----------
        pos : dict, optional
            A dictionary indicating the positional layout of the nodes. For more information visit
            `this page <https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw.html>`_
            By default is suitable will be generated.

        Examples
        --------

        We initialize a backtracking tree and visualize:

        ::

            from qrisp import auto_uncompute, QuantumBool, QuantumFloat
            from qrisp.quantum_backtracking import QuantumBacktrackingTree
            import matplotlib.pyplot as plt

            @auto_uncompute
            def reject(tree):
                return QuantumBool()

            @auto_uncompute
            def accept(tree):
                return (tree.branch_qa[0] == 1) & (tree.branch_qa[1] == 1) & (tree.branch_qa[2] == 1)


        >>> tree = QuantumBacktrackingTree(3, QuantumFloat(1, name = "branch_qf*"), accept, reject)
        >>> tree.init_node([])
        >>> tree.visualize_statevector()
        >>> plt.show()

        .. image:: ./root_state_plot.png
            :width: 200
            :alt: Root statevector plot
            :align: left

        >>> tree = tree.copy()
        >>> tree.init_phi([1,1,1])
        >>> tree.visualize_statevector()

        .. image:: ./phi_state_plot.png
            :width: 200
            :alt: Root statevector plot
            :align: left


        """

        G, root = self.statevector_graph(return_root = True)

        def tree_layout(G, node, depth, theta_parent, res_dic={}):

            r = depth + 1
            delta_theta = 2*np.pi/self.degree**(depth+1)
            theta_start = theta_parent - 2*np.pi/self.degree**(depth)/4

            children = list(G.neighbors(node))

            for i in range(len(children)):

                theta = theta_start + i*delta_theta

                res_dic[children[i]] = (r*np.sin(theta), r*np.cos(theta))

                tree_layout(G, children[i], depth+1, theta, res_dic)

            return res_dic


        pos= tree_layout(G, root, 0, 0)
        pos[root]= (0, 0)

        import colorsys
        def complex_to_color(cnumber):


            angle = np.angle(cnumber)
            radius = np.abs(cnumber)

            # Normalize the angle to the range [0, 2*pi)
            angle= (angle + np.pi*5/2) % (2 * np.pi)

            # Map the angle to the hue component of the color
            hue = angle / (2*np.pi)

            # Map the radius to the saturation and value components of the color
            saturation = 1
            value = 1

            # Convert the HSV components to RGB
            hsv_color = (hue, saturation, value)
            rgb_color = colorsys.hsv_to_rgb(*hsv_color)

            from scipy.special import expit

            intensity = expit((radius-0.2)*10)


            # Convert the RGB components to hexadecimal format
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb_color[0] * 255*intensity),
                int(rgb_color[1] * 255*intensity),
                int(rgb_color[2] * 255*intensity)
            )

            return hex_color

        colors = [complex_to_color(node.amplitude) for node in G.nodes()]

        nx.draw(G, pos)
        nx.draw_networkx_nodes(G, pos, node_color=colors)

        edge_labels = dict([((n1, n2), l)
                            for n1, n2, l in G.edges(data="label")])

        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)

    def statevector(self):
        """
        Returns a SymPy statevector object representing the state of the tree
        with decoded node labels.

        Returns
        -------
        state : sympy.Expr
            A SymPy quantum state representing the statevector of the tree.

        Examples
        --------

        We create a QuantumBacktrackingTree with and investigate the action of the
        :meth:`quantum step diffuser <qrisp.quantum_backtracking.QuantumBacktrackingTree.qstep_diffuser>`

        ::

            from qrisp import auto_uncompute, mcx, QuantumBool, QuantumFloat
            from qrisp.quantum_backtracking import QuantumBacktrackingTree

            @auto_uncompute
            def accept(tree):
                height_cond = (tree.h == 1) # The [0,1] node has height 1
                path_cond = QuantumBool()
                mcx(list(tree.branch_qa)[1:], path_cond, ctrl_state="10")
                return path_cond & height_cond

            @auto_uncompute
            def reject(tree):
                height_cond = (tree.h == 2) # The [1] node has height 2
                path_cond = QuantumBool()
                mcx(list(tree.branch_qa)[-1], path_cond, ctrl_state="1")
                return path_cond & height_cond


        Create tree and initialize a node where neither accept nor reject are True.

        >>> tree = QuantumBacktrackingTree(3, QuantumFloat(1, name = "branch_qf*"), accept, reject)
        >>> tree.init_node([0,0])

        Evaluate statevector

        >>> print(tree.statevector())
        1.0*|[0, 0]>
        >>> tree.qstep_diffuser(even = False)
        >>> print(tree.statevector())
        -0.666660010814667*|[0, 0, 0]> - 0.666660010814667*|[0, 0, 1]> + 0.333330005407333*|[0, 0]>

        We see that the :meth:`quantum step diffuser <qrisp.quantum_backtracking.QuantumBacktrackingTree.qstep_diffuser>`
        moves the state to the children of the [0,0] node (ie. [0,0,0] and [0,0,1]).

        We now investigate how it behaves on nodes that are accepted/rejected:

        Initiate a new tree

        >>> tree = tree.copy()
        >>> tree.init_node([0,1])
        >>> tree.qstep_diffuser(even = False)
        >>> tree.statevector()
        1.0*|[0, 1]>

        As expected, the accepted node is invariant.

        To investigate the rejected node, we create another copy:

        >>> tree = tree.copy()
        >>> tree.init_node([1])
        >>> tree.qstep_diffuser(even = True)
        >>> tree.statevector()
        -1*|[1]>

        As expected, the node has eigenvalue -1.

        If you are unsure why these statevectors are eigenvector please check
        `the paper <https://arxiv.org/abs/1509.02374>`_.

        """

        sv_function = self.qs.statevector("function", decimals = 10)

        # Internal qvs are the quantum variables that specify a backtrackingtree node
        # internal_qvs = [self.h, self.branch_workspace] + list(self.branch_qa)
        internal_qvs = [self.h] + list(self.branch_qa)
        # External qvs are any quantum variables that are also registered in the QuantumSession
        # but don't specify a node
        external_qvs = list(self.qs.qv_list)

        # Remove the internal qvs from the external qv_list
        for qv in internal_qvs:
            for i in range(len(external_qvs)):
                if hash(external_qvs[i]) == hash(qv):
                    external_qvs.pop(i)
                    break

        # Get a list of possible labels for each qv

        internal_qv_labels = []
        for qv in internal_qvs:
            label_list = []
            for i in range(2**qv.size):
                label_list.append(qv.decoder(i))
            internal_qv_labels.append(label_list)

        external_qv_labels = []
        for qv in external_qvs:
            label_list = []
            for i in range(2**qv.size):
                label_list.append(qv.decoder(i))
            external_qv_labels.append(label_list)

        # This retrieves a list of all possible constellations of labels
        internal_label_product = list(product(*internal_qv_labels))
        external_label_product = list(product(*external_qv_labels))

        # This will be the sympy object that is returned
        res_state = 0

        # Go through all internal label consteallations (ie. node states)
        for internal_label_const in internal_label_product[::-1]:

            # print(internal_label_const)
            # Get the path to that node state
            path = self.path_decoder(internal_label_const[0], internal_label_const[1:])
            # print(path)


            # Create a label dic for the sv_function
            internal_label_dic= {internal_qvs[i]: internal_label_const[i] for i in range(len(internal_qvs))}

            # If there are no external qvs, we can simply call the sv_function
            # with the label dic
            if len(external_qvs) == 0:

                amplitude = sv_function(internal_label_dic, 5)

                if abs(amplitude) < 1E-5:
                    continue

                # print(res_state)
                # Add the corresponding ket
                res_state += np.round(amplitude, 5) * OrthogonalKet(str(path))

                # print(res_state)


            # If there are external qvs we do a similar procedure to go through
            # all label constellations
            else:

                for external_label_const in external_label_product:


                    # Set up the external label dic
                    external_label_dic= {external_qvs[i]: external_label_const[i] for i in range(len(external_qvs))}

                    # Integrate the internal label dic
                    external_label_dic.update(internal_label_dic)

                    # Retrieve the amplitude
                    amplitude = sv_function(external_label_dic)

                    if abs(amplitude) < 1E-5:
                        continue

                    external_ket_expr = 1
                    # Generate the ket expression for the external qvs
                    for label in external_label_const:
                        external_ket_expr *= OrthogonalKet(label)

                    # Add the corresponding state
                    res_state += amplitude * \
                        OrthogonalKet(str(path)) * external_ket_expr

        return res_state


class OHQInt(QuantumVariable):

    def decoder(self, i):
        # One hot encoding:
        # Red:   [1,0,0,0]
        # Green: [0,1,0,0]
        # Yellow:[0,0,1,0]
        # Blue:  [0,0,0,1]

        # [0,0,0,1][0,0,1,0]

        # [0,0,0,0][1,0,0,1]
        is_power_of_two = ((i & (i-1) == 0) and i != 0)

        if is_power_of_two:
            return int(np.log2(i))

        else:
            return -3

    def __eq__(self, other):

        if isinstance(other, int):

            self.encoder(other)

            eq_qbl = QuantumBool()

            cx(self[other], eq_qbl)
            return eq_qbl

        else:
            raise Exception(
                f"Comparison with type {type(other)} not implemented")

    __hash__ = QuantumVariable.__hash__

    def is_even(self):
        is_even = QuantumBool()

        for i in range(self.size):
            if not i % 2:
                cx(self[i], is_even)


        return is_even

    def is_odd(self):
        is_odd = QuantumBool()

        for i in range(self.size):
            if i % 2:
                cx(self[i], is_odd)

        return is_odd
    
    def __lt__(self, other):
        
        if isinstance(other, int):
            less_than = QuantumBool()
            for i in range(self.size):
                if i < other:
                    cx(self[i], less_than)
            return less_than
        
        else:
            raise Exception(f"Comparison for type {type(other)} not implemented")



def fan_in(control, target):
    for qb in control:
        cx(control, target)





"""
This function realizes the operator U_x, which has the property

U_x |x> = |psi_x>

For more details on these objects, check the details of the
paper (https://arxiv.org/abs/1509.02374)or the beginning of this file.

The general idea to implement this operator are the following two steps:

    1. Manipulatre h such that |h> -> 1/N(|h> + c*|h-1>) with suitable N,c \in R
    2. Manipulate branch_qa controlled on h-1 to bring the new branches into
        into superposition.

The value of the constant c depends on wether x is the root or a terminal node.

For the root c = n**0.5*d_x**0.5
Otherwise c = d_x**0.5

Where d_x is the degree of the node.

"""
def psi_prep(x, even=True, min_height_assumption = 0):

    # Determine c
    c = x.degree**0.5

    # The step |h> -> 1/N(|h> + c*|h-1>) will be performed by a ry gate
    phase = np.arctan(c)*2
    root_phase = np.arctan((c*np.sqrt(x.max_depth)))*2

    rev_branch_qa = x.branch_qa
    # N = x.max_depth+1
    N = x.max_depth
    
    # To achieve the first step, we use a circuit, that performs a similar function
    # as a parametrized swap. That means:
    # |00> ==> |00>
    # |01> ==> sin(theta)*|01> + cos(theta)*|10>
    # |10> ==> cos(theta)*|10> + sin(theta)*|01>
    # |11> ==> |11>
    
    #This way we can "move" the 1 of the one hot encoding up and down.
    
    # After we moved the one, we apply an H-gate controlled on the target of the
    # moevement, to set up the super position in the branch_qa

    # Furthermore we also need to make sure that non-algorithmic states stay
    # invariant under the U_x (otherwise they will also get tagged in qstep_diffuser).
    
    # We achieve this by controlling the parameterized swap on the QuantumVariable
    # which would be set to super position by the controlled H-gate.
    # This QuantumVariable represents the branch information of the child states
    # of H_x = <{|y> | |x> -> |y>} U {|x>} >.
    # If it is in a non-zero state but the height variable indicates the parent
    # state, the state is invariant.
    
    if bool(N % 2) != even:
        c_iswap_reduced(root_phase, rev_branch_qa[N-1], x.h[N-1], x.h[N])
        x.qs.append(ch_gate, [x.h[N-1], rev_branch_qa[N-1]])

    for i in range(int(even), N-1, 2):
        if i + 1 < min_height_assumption:
            continue
        c_iswap_reduced(phase, rev_branch_qa[i], x.h[i], x.h[i+1])
        x.qs.append(ch_gate, [x.h[i], rev_branch_qa[i]])
    
    

ch_gate = HGate().control()

# This circuit is a slightly modified (and controlled) version of the XXPlusYY gate
# https://qiskit.org/documentation/stubs/qiskit.circuit.library.XXPlusYYGate.html
def c_iswap(phi, ctrl, target_1, target_0):
    h(target_1)
    cx(target_1, target_0)
    x(ctrl)
    ctrl.qs().append(RYGate(-phi/2).control(), [ctrl, target_0])
    ctrl.qs().append(RYGate(-phi/2).control(), [ctrl, target_1])
    x(ctrl)
    cx(target_1, target_0)
    h(target_1)

# This circuit performs a similar function as the previous one but has a different
# behavior on |11> and also requires less resources.
# Since the |11> behavior is irrelevant we can also use this circuit.

def c_iswap_reduced(phi, ctrl, target_0, target_1):
    
    phi = -phi/4 + np.pi/4
    h(target_1)
    cx(target_1, target_0)
    x(ctrl)
    ry(phi, target_0)
    ry(phi, target_1)
    
    #Usually we would now execute two controlled H-gates
    #If there are multiple controls, we'd have to perform
    #the corresponding mcx gate twice (to have two multi controlled H gates)
    
    # ctrl.qs().append(ch_gate, [ctrl, target_0])
    # ctrl.qs().append(ch_gate, [ctrl, target_1])
    
    #Two prevent this situation we execute a circuit with the same semantics but
    #only a single mcx gate
    
    #--------------------------
    
    #These one qubit gates make sure that the cx gates are acting as a controlled    
    #H-Gate 
    s([target_0, target_1])
    h([target_0, target_1])
    t([target_0, target_1])
    
    
    if len(ctrl) == 1:
        cx(ctrl, target_0)
        cx(ctrl, target_1)
    else:
        cx(target_0, target_1)
        mcx(ctrl, target_0)
        cx(target_0, target_1)
    
    t_dg([target_0, target_1])
    h([target_0, target_1])
    s_dg([target_0, target_1])
    
    #----------------------------
    ry(-phi, target_0)
    ry(-phi, target_1)
    x(ctrl)
    cx(target_1, target_0)
    h(target_1)



class Subtree(QuantumBacktrackingTree):

    def __init__(self, parent_tree, root_path):

        if len(root_path) > parent_tree.max_depth:
            raise Exception(
                "Tried to initialise subtree with root path longer than maximum depth")

        QuantumBacktrackingTree.__init__(self,
                                         parent_tree.max_depth,
                                         parent_tree.branch_qa[0],
                                         parent_tree.accept_function,
                                         parent_tree.reject_function,
                                         parent_tree.subspace_optimization
                                         )

        self.max_depth = parent_tree.max_depth - len(root_path)

        self.root_path = root_path
        self.original_tree = parent_tree

    def init_node(self, path):

        self.h[:] = self.max_depth - len(path)

        path = self.root_path + path

        if len(path):
            self.branch_qa[-len(path):] = path[::-1]

    def init_phi(self, path):

        h_state = {}
        h_state[self.max_depth] = (self.max_depth)**0.5

        for i in range(1, len(path)+1):
            h_state[self.max_depth - i] = (-1)**(i)

        self.h[:] = h_state

        rev_branch_qa = self.branch_qa[::-1]

        with self.h == self.max_depth:
            for k in range(len(self.root_path)):
                rev_branch_qa[k][:] = self.root_path[k]

        for i in range(1, len(path)+1):
            with self.h == self.max_depth - i:
                for j in range(i):
                    rev_branch_qa[j].encode(
                        path[j], permit_dirtyness=True)

                for k in range(len(self.root_path)):
                    rev_branch_qa[k+i][:] = self.root_path[k]

    def subtree(self, path):
        return self.original_tree.subtree(path)



def find_solution(tree, precision, cl_accept=None, traversed_nodes=None, measurement_kwargs={}):
    # The idea of this function is to use the quantum algorithm to check wether
    # a the subtree of a given node contains a solution and then recursively call
    # this function on that subtree.


    # If there is no classical accept function given, we create a copy of the original
    # tree and evaluate the quantum accept function on that node via the simulator
    if cl_accept is None:
        def cl_accept(path):
            if isinstance(tree, Subtree):
                copied_tree = tree.original_tree.copy()
            else:
                copied_tree = tree.copy()
            copied_tree.init_node(path)
            accept_qbl = copied_tree.accept()
            mes_res = accept_qbl.get_measurement()
            return mes_res == {True: 1}


    # The first step is to check wether the current root is a solution
    if isinstance(tree, Subtree):
        path = tree.root_path
    else:
        path = []

    if cl_accept(path):
        return path
    elif tree.max_depth == 0:
        return None

    # This list keeps track of which nodes have already been checked for solutions
    if traversed_nodes is None:
        traversed_nodes = []

    # Initialize the root node
    tree.init_node([])

    # Perform quantum phase estimation
    qpe_res = tree.estimate_phase(precision)

    # Retrieve the measurement results
    mes_res = multi_measurement([qpe_res, tree.h, tree.branch_qa], **measurement_kwargs)
    
    # We will first check wether there is a solution
    # The s variable will contain the probability to measure
    # the qpe_res == 0 branch.
    s = 0
    # This list will contain the possible branches
    new_branches = []

    for k, v in mes_res.items():
        # k[0] is the value of qpe_res
        if k[0] == 0:
            # If the measurement result is part of the qpe == 0 branch, add the
            # probability to s
            s += v

            new_branches.append(k)
    # If the probability is less then 0.25, there is no solution
    if s <= 0.25:
        return None

    # If the probability is between 0.25 and 0.375, the qpe needs more precision
    if s <= 0.375:
        raise Exception(
            "Executed find solution method of quantum backtracking algorithm with insufficient precision")


    # To find the next node to check we will use a heuristic.
    # After measurement of the 0 branch, the tree is collapsed to a state which
    # also contains the |phi> state. The phi state is a superposition of nodes
    # leading to the desired solution.
    # We assume that the outcome with the smallest tree.h value is the state
    # corresponding to |phi>. This proved to be the case in every situation we tested.
    # If this assumption for whichever reason is not correct, the solution will still
    # be found because of the recursive nature of this algorithm.

    # Sort for the value of tree.h
    new_branches.sort(key=lambda x: x[1])
    for b in new_branches:
        
        # Get the path to the new node
        if isinstance(tree, Subtree):
            new_path=tree.original_tree.path_decoder(b[1], b[2])
        else:
            new_path=tree.path_decoder(b[1], b[2])

        # Continue if new_path was already explored 
        if tuple(new_path) in traversed_nodes or tuple(new_path)==tuple(path):
            continue 

        # Generate the subtree
        subtree=tree.subtree(new_path)

        # Recursive call
        solution=find_solution(subtree, precision, cl_accept, traversed_nodes, measurement_kwargs=measurement_kwargs)

        # Leave loop if solution was found
        if solution is not None:
            break
        else:
            traversed_nodes.append(tuple(new_path))

    else:
        raise Exception(
            "Executed find solution method of quantum backtracking algorithm with insufficient precision")

    return solution



class QBTNode:

    def __init__(self, tree, path, amplitude=None):

        self.h=tree.max_depth - len(path)
        self.path=path
        self.tree=tree
        self.amplitude=amplitude

    def __hash__(self):
        return hash(str(self.path))

    def sv_specifier(self):
        amplitude_state_specifyer={
            self.tree.h: self.tree.max_depth - len(self.path)}

        path=list(self.path)
        if isinstance(self.tree, Subtree):
            path=self.tree.root_path + path

        for k in range(len(self.tree.branch_qa)):
            if k < len(path):
                amplitude_state_specifyer[self.tree.branch_qa[-1-k]]=path[k]
            else:
                amplitude_state_specifyer[self.tree.branch_qa[-1-k]]=0

        return amplitude_state_specifyer

    def __str__(self):
        return "QBTNode(path = " + str(self.path) + ", amplitude = " + str(self.amplitude) + ")"

    def __repr__(self):
        return str(self)
