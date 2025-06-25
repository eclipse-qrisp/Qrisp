.. _ft_compilation:

Fault-Tolerant compilation
==========================

As quantum computing `steadily progresses towards the era of fault-tolerant quantum processors <https://www.quera.com/press-releases/harvard-quera-mit-and-the-nist-university-of-maryland-usher-in-new-era-of-quantum-computing-by-performing-complex-error-corrected-quantum-algorithms-on-48-logical-qubits>`_, the landscape of compilation techniques undergoes a profound transformation. While Noisy Intermediate-Scale Quantum (NISQ) devices have been the pioneers in demonstrating quantum computational capabilities, the vision for practical quantum computing hinges on fault-tolerant architectures capable of handling errors that naturally arise in quantum systems.

This tutorial delves into the realm of compiling for fault-tolerant quantum devices, exploring the specialized techniques and considerations that set this stage apart from the compilation challenges encountered in NISQ environments. In the quest for fault tolerance, quantum error correction becomes a paramount concern, demanding innovative strategies to mitigate errors and maintain the integrity of quantum computations over extended periods.

At the end of the tutorial we will demonstrate how the previously coded Shor algorithm can be compiled to peak performance, so stay tuned!

T-Gates
-------

One of the major differences in compiling for FT devices is the fact that CNOT gates are no longer the bottleneck they used to be for NISQ devices. The new bottleneck is the execution of $T$-gates since these gates require `distillation of magic states  <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.020341>`_ and subsequent teleportation into the quantum computer. Qrisp provides a variety of tools to optimize the amount of T-gates as well as their depth.

Evaluating performance
----------------------

The first thing you need to know is how to *measure* these quantities for your Qrisp algorithms.

For the T-gate count this is relatively similar to the way it is done for the NISQ case. To demonstrate, we compile a :ref:`QuantumCircuit` and measure the T-count.

>>> from qrisp import QuantumVariable, QuantumBool, x, cx, t, rz
>>> a = QuantumVariable(2)
>>> b = QuantumVariable(2)
>>> x(a[0])
>>> t(a[0])
>>> cx(a[0], b[0])
>>> t(a[1])
>>> t(b[0])
>>> qc = a.qs.compile()
>>> print(qc)
     ┌───┐┌───┐          
a.0: ┤ X ├┤ T ├──■───────
     ├───┤└───┘  │       
a.1: ┤ T ├───────┼───────
     └───┘     ┌─┴─┐┌───┐
b.0: ──────────┤ X ├┤ T ├
               └───┘└───┘
b.1: ────────────────────
                                             
We now count the amount of T-gates using the :meth:`count_ops <qrisp.QuantumCircuit.count_ops>` method and measure the T-depth using the :meth:`t_depth <qrisp.QuantumCircuit.t_depth>` method.

.. note:: As you might have noticed, T-depth and T-count are two different things. The former is approximately proportional to the **maximum speed** the algorithm can be executed with. The latter is the amount of T-gates. Please note that as of right now it's unclear how efficient magic-state destillation will happen. Therefore, a future quantum computer, that can create magic states efficiently will be bottlenecked by the T-depth but a device where this process takes a lot of time will be bottlenecked by that T-count.

>>> qc.count_ops()
{'x': 1, 't': 3, 'cx': 1}
>>> qc.t_depth()
2

For the T-depth we get the value 2 because the qubit ``b.0`` has to wait until ``a.0`` is finished.
Many applications require more than the Clifford + T gate set. Especially parametrized Pauli-rotations are used often in variational algorithms or hamiltonian simulations. To execute the parametrized Pauli-rotations a `synthesis algorithm <https://arxiv.org/abs/1403.2975>`__ needs to be called, which approximates the required rotation using only Clifford + T gates. According to the linked paper, this requires $3\log_2(\frac{1}{\epsilon})$ T-gates, where $\epsilon$ is the desired precision. Qrisp accounts for this effort if given the desired precision $\epsilon$.

>>> rz(np.pi/8, b[0])
>>> qc = a.qs.compile()
>>> qc.t_depth(epsilon = 2**-3)
11.0
>>> print(qc)
     ┌───┐┌───┐                     
a.0: ┤ X ├┤ T ├──■──────────────────
     ├───┤└───┘  │                  
a.1: ┤ T ├───────┼──────────────────
     └───┘     ┌─┴─┐┌───┐┌─────────┐
b.0: ──────────┤ X ├┤ T ├┤ Rz(π/8) ├
               └───┘└───┘└─────────┘
b.1: ───────────────────────────────     

If no $\epsilon$ is given, Qrisp will search for the smallest angle $\phi_{min}$ in the circuit and set $\epsilon = \phi_{min}/8$. Please note that we could find close to no research on how to automatically determine a reasonable value for $\epsilon$ so this is a tentative solution.

Improving performance
---------------------

The first and foremost performance enhancement that Qrisp achieves is gate-speed aware compilation. For a detailed description please read this section :ref:`compile <gate_speed_aware_comp>`. This feature allows the Qrisp compiler to modify the order of gates and improve the allocation mechanism under the consideration that certain gates take more time than others. The result is that in a physical circuit execution, the overal run-time is significantly enhanced.
To specify the gate-speed of the backend you are targeting, you can write your own gate-speed function. This function should take an :ref:`Operation` object and return the corresponding speed.

For instance:
::

    def toy_gate_speed(op):
    
        if op.name == "x":
            return 1
        if op.name == "y":
            return 10
        else:
            return 0
      
            
This gate-speed function describes a rather exotic backend where the X-gate takes 1 time unit (for instance nanoseconds), the Y-gate 10 time units and every other gate is executed instantaneously. To specify the T-gate speed you can use the built-in function: :meth:`t_depth_indicator <qrisp.t_depth_indicator>`. This function assigns $T$, $T^\dagger$ and $P(\pm \frac{\pi}{4})$ gates a speed of 1, parametrized gates (such as the above) a gate-speed of $3\log_2(\frac{1}{\epsilon})$ and every clifford gate a time of 0.

::

    from qrisp import t_depth_indicator

    def gate_speed(op):
        return t_depth_indicator(op, epsilon = 2**-3)


We can now use this function to inform the compiler about the gate-speed:

>>> qc = a.qs.compile(gate_speed = gate_speed)
>>> qc.t_depth(epsilon = 2**-3)
10.0
>>> print(qc)
     ┌───┐     ┌───┐           
a.0: ┤ X ├──■──┤ T ├───────────
     ├───┤  │  └───┘           
a.1: ┤ T ├──┼──────────────────
     └───┘┌─┴─┐┌───┐┌─────────┐
b.0: ─────┤ X ├┤ T ├┤ Rz(π/8) ├
          └───┘└───┘└─────────┘
b.1: ──────────────────────────
                               
            
We see how the compiler moved the T-gate on ``a.0`` after the CNOT gate, so the follow up on ``b.0`` can be executed in parallel with ``a.0``, thereby reducing the required T-depth.

Fault-Tolerant Toffolis
-----------------------

Another important important leverage Qrisp offers for fault-tolerant compilation is the use of specialized :meth:`mcx<qrisp.mcx>` implementations. The most relevant methods here are ``jones`` and ``gidney``. The default Toffoli implementation (requiring no ancillae) requires a T-depth of 4.

>>> from qrisp import mcx
>>> ctrl = QuantumVariable(2)
>>> target = QuantumBool()
>>> mcx(ctrl, target)
>>> ctrl.qs.compile().t_depth()
4

With the ``jones`` method this is reduced significantly:

>>> ctrl = QuantumVariable(2)
>>> target = QuantumBool()
>>> mcx(ctrl, target, method = "jones")
>>> ctrl.qs.compile(compile_mcm = True).t_depth()
1

How is this possible?! Well first of all, the ``jones`` method requires two ancilla qubits - this news isn't as bad as it seems, because the Qrisp compiler automatically reuses the qubits from previously deallocated variables, so in many practical situations the qubit overhead is 0. The other important point is that this Toffoli technique uses a mid-circuit measurement with a subsequent classically controlled CZ-gate. For more details, visit the `publication <https://arxiv.org/abs/1212.5069>`_. Note that the measurement is not inserted until the ``compile`` method is called - before that, a representative is used. This allows to still query the :meth:`statevector <qrisp.QuantumSession.statevector>` simulator (even though the final circuit contains a measurement) and thus significantly simplifies debugging. Note that you can activate/deactivate the compilation of mid-circuit measurements using the ``compile_mcm`` keyword.

The second relevant Toffoli method is ``gidney``. `Gidney`s temporary logical AND <https://arxiv.org/abs/1709.06648>`_ always comes im pairs - one computation and one uncomputation. If you know that your algorithm contains a pattern like this, the ``gidney`` method is a very good option for you. This is because the computation part (similar to ``jones``) has a T-depth of 1 (asterisk!)
but the uncomputation part has a T-depth of 0! Basically this implies **another** $\times 2$ speed-up for cases where this structure is applicable.

In Qrisp, you can perform the computation as you normally would.

>>> ctrl = QuantumVariable(2)
>>> target = QuantumBool()
>>> mcx(ctrl, target, method = "gidney")

To call the uncomputation part, you can use the :meth:`uncompute <qrisp.QuantumVariable.uncompute>` method or call it within an :ref:`InversionEnvironment`.

>>> target.uncompute()

or

>>> with invert(): mcx(ctrl, target, method = "gidney")
>>> ctrl.qs.compile(compile_mcm = True).t_depth()
2

Note that the T-depth yields 2 here, but the delay in many applications will still be 1. To understand why, consider that the control qubits most likely will already have gone through some non-clifford gates, while the target is usually freshly allocated. To demonstrate we execute some T-gates on ``ctrl`` before the Toffoli gate is applied:

>>> ctrl = QuantumVariable(2)
>>> t(ctrl)
>>> t(ctrl)
>>> target = QuantumBool()
>>> mcx(ctrl, target, method = "gidney")
>>> print(ctrl.qs.compile(compile_mcm = True).transpile())
          ┌───┐┌───┐          ┌───┐┌─────┐       ┌───┐               
  ctrl.0: ┤ T ├┤ T ├──■───────┤ X ├┤ Tdg ├───────┤ X ├───────────────
          ├───┤├───┤  │       └─┬─┘└┬───┬┘┌─────┐└─┬─┘┌───┐          
  ctrl.1: ┤ T ├┤ T ├──┼────■────┼───┤ X ├─┤ Tdg ├──┼──┤ X ├──────────
          ├───┤├───┤┌─┴─┐┌─┴─┐  │   └─┬─┘ └┬───┬┘  │  └─┬─┘┌───┐┌───┐
target.0: ┤ H ├┤ T ├┤ X ├┤ X ├──■─────■────┤ T ├───■────■──┤ H ├┤ S ├
          └───┘└───┘└───┘└───┘             └───┘           └───┘└───┘
          
As we see, the first T-gate on ``target.0`` can be executed while the T-gates of ``ctrl`` are still running.

.. _ft_compilation_shor:

Fault tolerant compilation of Shor's algorithm
----------------------------------------------

In the :ref:`previous tutorial<shor_tutorial>`, you learned how to implement Shor's algorithm. Obviously the things you learned in the above section all apply to compile it for fault-tolerant backends, but the most powerfull optimization still awaits!

Selecting a suited adder
^^^^^^^^^^^^^^^^^^^^^^^^

The Qrisp implementation of Shor's algorithm allows you to provide an arbitrary adder for the execution of the required arithmetic. We provide some pre-implemented adders notably:

* The :meth:`fourier_adder <qrisp.fourier_adder>` (`paper <https://arxiv.org/abs/quant-ph/0008033>`__) requires minimal qubit overhead and has a very efficient :meth:`custom_control <qrisp.custom_control>` but uses a lot of parametized phase gates, which increases the T-depth. The low qubit count makes it suitable for simulation, which is why it is the default adder.

* The :meth:`cucarro_adder <qrisp.cuccaro_adder>` (`paper <https://arxiv.org/abs/quant-ph/0410184>`__) also requires minimal qubits but no parametrized phase gates. It doesn't have a custom controlled version.

* The :meth:`gidney_adder <qrisp.gidney_adder>` (`paper <https://arxiv.org/abs/1709.06648>`__) requires $n$ ancillae but uses the ``gidney`` Toffoli method described above, making it very fast in terms of T-depth but also economical in terms of T-count.

* The :meth:`qcla <qrisp.qcla>` (`paper <https://arxiv.org/abs/2304.02921>`__) requires quite a lot of ancillae but has only logarithmic scaling when it comes to T-depth. It is faster than the Gidney adder for any input size larger than 7.

In general you can also write your own adder and try it out! Feel free to use the :meth:`inpl_adder_test <qrisp.inpl_adder_test>` function to verify your adder works.

To illustrate the difference, we benchmark the :meth:`gidney_adder <qrisp.gidney_adder>` vs. the :meth:`qcla <qrisp.qcla>` on the operation that is most relevant for Shor's algorithm: Controlled modular in-place multiplication.

::

    from qrisp import *
    N = 3295
    qg = QuantumModulus(N, inpl_adder = gidney_adder)
    
    ctrl_qbl = QuantumBool()
    
    with control(ctrl_qbl):
        qg *= 953
        
    gate_speed = lambda op : t_depth_indicator(op, epsilon = 2**-10)
     
    qc = qg.qs.compile(gate_speed = gate_speed, compile_mcm = True)
    print(qc.t_depth())
    # Yields 956
    print(qc.num_qubits())
    # Yields 79    
    
    
Now the :meth:`qcla <qrisp.qcla>`:

::

    qg = QuantumModulus(N, inpl_adder = qcla)
    
    ctrl_qbl = QuantumBool()
    
    with control(ctrl_qbl):
        qg *= 10
        
    qc = qg.qs.compile(workspace = 10, gate_speed = gate_speed, compile_mcm = True)
    
    print(qc.t_depth())s
    # Yields 784
    print(qc.num_qubits())
    # Yields 88   

We see that the T-depth is reduced by $\approx 20 \%$. Due to the logarithmic scaling of the adder, larger scales will profit even more! Note that we granted the compiler 10 qubits of :ref:`workspace <workspace>`, as this adder can profit a lot from this resource.

.. _adder_based_qft:

Addition based QFT
^^^^^^^^^^^^^^^^^^

If you made it this far, you probably heard about the :ref:`quantum fourier transform <QFT>` algorithm and it's circuit. This circuit contains a variety of parametrized phase gates, which can throttle the efficiency of a fault-tolerant backend as they have to be synthesized using T-gates. Fortunately a lot of this overhead can be remedied with a trick introduced in `this paper <https://arxiv.org/abs/2203.07739>`__. The observation here is that the QFT circuit contains repeated blocks of phase gates of the following structure.

::
                
          ┌─────────┐
    qb_0: ┤ P(π/16) ├
          └┬────────┤
    qb_1: ─┤ P(π/8) ├
           ├────────┤
    qb_2: ─┤ P(π/4) ├
           ├────────┤
    qb_3: ─┤ P(π/2) ├
           └────────┘

Acting on a quantum state $\ket{x}$ this produces the phase-shift

.. math::

    \ket{x} \rightarrow \text{exp}\left(\frac{i\pi x}{16}\right)\ket{x}
    
The idea is now to realize this kind of transformation using an adder, acting on a "reservoir" state. The reservoir is initialized once and can be used throughout all of the algorithm. This is because it is constructed such that it is an eigenstate under (modular) addition and produces the desired phase.

.. math::
    
    \ket{R} = \frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n-1} \text{exp}\left(\frac{-i \pi k}{2^n}\right) \ket{k}

If we now perform an addition on this state we get

.. math::
    
    \begin{align}
    U_{add}\ket{x}\ket{R} &= \frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n-1} \text{exp}\left(\frac{-i \pi k}{2^n}\right) U_{add} \ket{x} \ket{k}\\
    &= \frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n-1} \text{exp}\left(\frac{-i \pi k}{2^n}\right) \ket{x}\ket{k + x}\\
    &= \frac{1}{\sqrt{2^n}} \sum_{k=0}^{2^n-1} \text{exp}\left(\frac{-i \pi (k-x)}{2^n} \right)\ket{x} \ket{k}\\
    &= \text{exp}\left(\frac{i \pi x}{2^n}\right)\ket{x}\ket{R}
    \end{align}

In the third line we used a relabeling of the summation indices to move $x$ into the exponent.
We can therefore produce the above phase-shift using an adder of our choice. We spare you the details implementing this in the QFT algorithm (if you are interested please check the paper) and just tell you how you can use it:

>>> qf = QuantumFloat(4)
>>> qf[:] = 4
>>> QFT(qf, inpl_adder = gidney_adder)
>>> qc = qf.qs.compile(gate_speed = gate_speed, compile_mcm = True)
>>> print(qc.t_depth())
73.0

To verify that our construction is correct, we perform the inverse (regular) QFT:

>>> QFT(qf, inv = True)
>>> print(qf)
{4: 1.0}


This concludes our tutorial on fault-tolerant compilation. We hope you could gain some insights on what is possible with Qrisp and look forward to see your algorithms built with these tools! 🛠️
