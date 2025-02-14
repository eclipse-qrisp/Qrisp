.. _HHL_tutorial:

Solving linear system problems with HHL
=======================================

The Harrow-Hassidim-Lloyd (HHL) quantum algorithm offers an exponential speed-up over classical methods for solving linear system problems $Ax=b$ for certain sparse matrices $A$. 

The implementation features hybrid quantum-classical workflows and is compiled using `Catalyst <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`_, a quantum just-in-time (:ref:`QJIT <qjit>`) compiler framework.

The goal of this demo is to showcase how Qrisp and Catalyst complement each other for implemententing advanced quantum algorithms and compling them for practically relevant problem sizes through the :ref:`Jasp compilation pipeline <Jasp>`.

HHL algorithm in theory
-----------------------

Given an $N$-by-$N$ Hermitian matrix $A$ and an $N$-dimensional vector $b$, the Quantum Linear Systems Problem (QSLP) consists of preparing a quantum state $\ket{x}$ with amplitudes proportional to the solution $x$ of the linear system of equations $Ax=b$. 
Thereby, it can exhibit an exponential speedup over classical methods for certain sparse matrices $A$. The HHL quantum algorithm and, more generally, quantum linear systems algorithms, hold significant promise for accelerating computations in fields that rely 
heavily on solving linear systems of equations, such as `solving differential equations <https://arxiv.org/abs/2202.01054v4>`_, or acclerating machine learning.

In its eigenbasis, the matrix $A$ can be written as 
$$ A = \\sum_i \\lambda_i\\ket{u_i}\\bra{u_i} $$
where $\ket{u_i}$ is an eigenvector of $A$ corresponding to the eigenvalue $\lambda_i$.

We define the quantum states $\ket{b}$ and $\ket{x}$ as
$$ \\ket{b} = \\dfrac{\\sum_i b_i\\ket{i}}{\\|\\sum_i b_i\\ket{i}\\|} = \\sum_i \\beta_i\\ket{u_i} \\quad\\text{and}\\quad \\ket{x} = \\dfrac{\\sum_i x_i\\ket{i}}{\\|\\sum_i x_i\\ket{i}\\|} = \\sum_i \\gamma_i\\ket{u_i} $$
where $\ket{b}$ and $\ket{x}$ are expressed in the eigenbasis of $A$.

Solving the linerar system amounts to
$$\\begin{align}\\ket{x}&=A^{-1}\\ket{b}\\\\&=\\bigg(\\sum_{i=0}^{N-1}\\lambda_i^{-1}\\ket{u_i}\\bra{u_i}\\bigg)\\sum_j\\beta_j\\ket{u_j}\\\\&=\\sum_{i=0}^{N-1}\\lambda_i^{-1}\\beta_i\\ket{u_i}\\end{align}$$

You might wonder why we can't just apply $A^{-1}$ directly to $\ket{b}$? This is because, in general, the matix $A$ is not unitary. 
However, we will circumnavigate this by exploiting that the Hamiltonian evolution $U=e^{itA}$ is unitary for a Hemitian matrix $A$. And this brings us to the HHL algorithm.

In theory, the HHL algorithm can be described as follows:

* Step 1: We start by :ref:`preparing <prepare>` the state 
  $$ \\ket{\\Psi_1} = \\ket{b} = \\sum_i \\beta_i\\ket{u_i}$$

* Step 2: Applying :ref:`quantum phase estimation <QPE>` with respect to the Hamiltonian evolution $U=e^{itA}$ yields the state 
  $$ \\ket{\\Psi_2} = \\sum_i \\beta_i\\ket{u_i}\\ket{\\lambda_jt/2\\pi} = \\sum_i \\beta_i\\ket{u_i}\\ket{\\widetilde{\\lambda_i}} $$ 
  To simplify notation, we write $\widetilde{\lambda}_i=\lambda_jt/2\pi$.
  

* Step 3: Performing the inversion of the eigenvalues $\widetilde{\lambda}_i\rightarrow\widetilde{\lambda}_i^{-1}$ yields the state
  $$ \\ket{\\Psi_3} = \\sum_i \\beta_i\\ket{u_i}\\ket{\\widetilde{\\lambda_i}}\\ket{\\widetilde{\\lambda_i^{-1}}} $$

* Step 4: The amplitudes are multiplied by the inverse eigenvalues $\widetilde{\lambda}_i^{-1}$ to obtain the state
  $$ \\ket{\\Psi_4} = \\sum_i \\lambda_i^{-1}\\beta_i\\ket{u_i}\\ket{\\widetilde{\\lambda}_i}\\ket{\\widetilde{\\lambda}_i^{-1}} $$
  This is achieved by means of a repeat-until-success procedure that applies **Steps 1-3** as a subroutine. Stay tuned for more details below!


* Step 5: As a final step, we uncompute the variables $\ket{\widetilde{\lambda}^{-1}}$ and $\ket{\widetilde{\lambda}}$, and obtain the state
  $$ \\ket{\\Psi_5} = \\sum_i \\lambda_i^{-1}\\beta_i\\ket{u_i} = \\ket{x} $$

And that's the HHL algorithm. The variable initialized in state $\ket{b}$ is now found in state $\ket{x}$! 

As shown in the `original paper <https://arxiv.org/pdf/0811.3171>`_, the runtime of this algorithm is $\mathcal{O}(\log(N)s^2\kappa^2/\epsilon)$ 
where $s$ and $\kappa$ are the sparsity and condition number of the matrix $A$, respectively, and $\epsilon$ is the precison of the solution. The logarithmic dependence on the dimension $N$ is the source of an exponential advantage over classical methods.

HHL implementation in practice
------------------------------

Let's put theory into practice and dive into an implementation of the HHL algorithm in 

As a fist step, we define a function ``fake_inversion`` that performs the inversion $\lambda\mapsto\lambda^{-1}$. In this example, we restict ourselves to an implementation that works for values $\lambda=2^{-k}$ for $k\in\mathbb N$.
(As shown above, a general :ref:`inversion <InversionEnvironment>` is available in Qrisp and will soon be updated to be compatible with QJIT compilation!)

::
    
    from qrisp import *

    def fake_inversion(qf, res=None):
        if res is None:
            res = QuantumFloat(qf.size+1)

        for i in jrange(qf.size):
            cx(qf[i],res[qf.size-i])

        return res

                                             
Essentially, the controlled-NOT operations in the loop reverse the positions of the bits in input variable and place them in the result variable in the opposite order. 
For example, for $\lambda=2^{-3}$, which is $0.001$ in binary, the function would produce $\lambda^{-1}=2^3$, which in binary is 1000.

Let's see if it works as intended!

::

    qf = QuantumFloat(3,-3)
    x(qf[2])
    dicke_state(qf, 1)
    res = fake_inversion(qf)
    print(multi_measurement([qf, res]))


Next, we define the function ``HHL_encoding`` that performs **Steps 1-4** and prepares the state $\ket{\Psi_4}$.
But, how do get the values $\widetilde{\lambda}^{-1}_i$ into the amplitudes of the states, i.e. how do we go from $\ket{\Psi_3}$ to $\ket{\Psi_4}$?

Recently, efficient methods for black-box quantum state preparation that avoid arithmetic were proposed, see `Sanders et al. <https://arxiv.org/pdf/1807.03206>`_, `Wang et al. <https://arxiv.org/pdf/2012.11056>`_ In this demo, we use a routine proposed in the latter reference which is based on a comparison between integers. This is implemented via the aforementioned comparisons of QuantumFloats.

To simplify the notation, we write $y^{(i)}=\widetilde{\lambda}^{-1}_i$. Recall that the values $y^{(i)}$ represent unsigned integers between $0$ and $2^n-1$. 

Starting from the state
$$ \\ket{\\Psi_3} = \\sum_i \\beta_i\\ket{u_i}\\ket{\\widetilde{\\lambda_i}}\\ket{y^{(i)}}_{\\text{res}} $$

we prepare a uniform superposition of $2^n$ states in a ``case_indicator`` QuantumFloat.
$$ \\ket{\\Psi_3'} = \\sum_i \\beta_i\\ket{u_i}\\ket{\\widetilde{\\lambda_i}}\\ket{y^{(i)}}_{\\text{res}}\\otimes\\frac{1}{\\sqrt{2^n}}\\sum_{x=0}^{2^n-1}\\ket{x}_{\\text{case}} $$

Next we calculate the comparison $a\geq b$ between the ``res`` and the ``case_indicator`` into a QuantumBool ``qbl``.
$$ \\ket{\\Psi_3''} = \\sum_i \\beta_i\\ket{u_i}\\ket{\\widetilde{\\lambda_i}}\\ket{y^{(i)}}_{\\text{res}}\\otimes\\frac{1}{\\sqrt{2^n}}\\left(\\sum_{x=0}^{y^{(i)}-1}\\ket{x}_{\\text{case}}\\ket{0}_{\\text{qbl}} + \\sum_{x=y^{(i)}}^{2^n-1}\\ket{x}_{\\text{case}}\\ket{1}_{\\text{qbl}}\\right) $$

Finally, the ``case_indicator`` is unprepared with $n$ Hadamards and we obtain the state
$$ \\ket{\\Psi_3'''} = \\sum_i \\dfrac{y^{(i)}}{2^n}\\beta_i\\ket{u_i}\\ket{\\widetilde{\\lambda_i}}\\ket{y^{(i)}}_{\\text{res}}\\ket{0}_{\\text{case}}\\ket{0}_{\\text{qbl}} + \\ket{\\Phi} $$

where $\ket{\Phi}$ is an orthogonal state with the last variables not in $\ket{0}_{\text{case}}\ket{0}_{\text{qbl}}$.

Hence, upon measuring the ``case_indicator`` in state $\ket{0}$ and the target ``qbl`` in state $\ket{0}$, the desired state is prepared. 

**Steps 1-4** are preformed as :ref:`repeat-until-success (RUS) routine <RUS>`. This decorator converts the function to be executed within a repeat-until-success (RUS) procedure. The function must return a boolean value as first return value and is repeatedly executed until the first return value is True.

::

    @RUS(static_argnums = [0,1])
    def HHL_encoding(b, hamiltonian_evolution, n, precision):

        # Prepare the state |b>. Step 1
        qf = QuantumFloat(n)
        # Reverse the endianness for compatibility with Hamiltonian simulation.
        prepare(qf, b, reversed=True)

        qpe_res = QPE(qf, hamiltonian_evolution, precision=precision) # Step 2
        inv_res = fake_inversion(qpe_res) # Step 3

        case_indicator = QuantumFloat(inv_res.size)

        with conjugate(qrisp.h)(case_indicator):
            qbl = (case_indicator >= inv_res)
        
        cancelation_bool = (measure(case_indicator) == 0) & (measure(qbl) == 0)

        # The irst return value is a boolean. Additional return values are QuantumVaraibles.
        return cancelation_bool, qf, qpe_res, inv_res

      
            
The probability of success could be further increased by oblivious :ref:`amplitude amplification<AA>` in order to obain an optimal asymptotic scaling.

Finally, we put all things together into the **HHL** function.

This function takes the follwoing arguments:

* ``b`` The vector $b$.
* ``hamiltonian_evolution`` A function performing hamiltonian_evolution $e^{itA}$.
* ``n`` The number of qubits encoding the state $\ket{b}$ ($N=2^n$).
* ``precision`` The precison of the quantum phase estimation.

The HHL function uses the previously defined subroutine to prepare the state $\ket{\Psi_4}$ and subsequently uncomputes the $\ket{\widetilde{\lambda}}$ and $\ket{\lambda}$ quantum variables leaving the first variable, 
that was initialized in state $\ket{b}$, in the target state $\ket{x}$.

::

    def HHL(b, hamiltonian_evolution, n, precision):

    qf, qpe_res, inv_res = HHL_encoding(b, hamiltonian_evolution, n, precision)
    
    with invert():
        QPE(qf, hamiltonian_evolution, target=qpe_res)
        fake_inversion(qpe_res, res=inv_res)

    # Reverse the endianness for compatibility with Hamiltonian simulation.
    for i in jrange(qf.size//2):
        swap(qf[i],qf[n-i-1])
    
    return qf


Applying HHL to solve systems of linear equations
-------------------------------------------------

Let's try a first simple example. First, the matrix $A$ is repesented as a Pauli operator $H$ and the Hamiltonian evolution unitary $U=e^{itH}$ is obtained by :meth:`.trotterization <qrisp.operators.qubit.QubitOperator.trotterization>` with 1 step 
(as the Pauli terms commute in this case). We choose $t=\pi$ to ensure that $\widetilde{\lambda}_i=\lambda_i t/2\pi$ are of the form $2^{-k}$ for a positive integer $k$.

This is enabled by the Qrisp's :ref:`QubitOperator <QubitOperator>` class providing the tools to describe, optimize and efficiently simulate quantum Hamiltonians.

::

    from qrisp.operators import QubitOperator
    import numpy as np

    A = np.array([[3/8, 1/8], 
                  [1/8, 3/8]])

    b = np.array([1,1])

    H = QubitOperator.from_matrix(A).to_pauli()

    # By default e^{-itH} is performed. Therefore, we set t=-pi.
    def U(qf):
        H.trotterization()(qf,t=-np.pi,steps=1)


The :ref:`terminal_sampling decorator <terminal_sampling>` performs a hybrid simulation and afterwards samples from the resulting quantum state. We convert the resulting measurement probabilities 
to amplitudes by appling the square root. Note that, minus signs of amplitudes cannot be recovered from measurement probabilities.

::

    @terminal_sampling
    def main():

        x = HHL(tuple(b), U, 1, 3)
        return x

    res_dict = main()

    for k, v in res_dict.items():
        res_dict[k] = v**0.5

    print(res_dict)


Finally, let's compare to the classical result. 

::

    x = (np.linalg.inv(A)@b)/np.linalg.norm(np.linalg.inv(A)@b)
    print(x)


And viola! Now, let's tackle some more complicated examples! Next, we try some randomly generated matrices whose eigenvalues are inverse powers of 2, i.e. of the form $2^{-k}$ for $k<K$.

To facilitate fast simulations, we restrict ourselves to $K=4$ (required ``precision`` of QPE) as the runtime of the HHL algorithm scales linearly in the inverse precision $\epsilon=2^{-K}$ (and therefore exponentially in $K$).

::

    def hermitian_matrix_with_power_of_2_eigenvalues(n):
        # Generate eigenvalues as inverse powers of 2.
        eigenvalues = 1/np.exp2(np.random.randint(1, 4, size=n))
        
        # Generate a random unitary matrix.
        Q, _ = np.linalg.qr(np.random.randn(n, n))
        
        # Construct the Hermitian matrix.
        A = Q @ np.diag(eigenvalues) @ Q.conj().T
        
        return A

    # Example 
    n = 3
    A = hermitian_matrix_with_power_of_2_eigenvalues(2**n)

    H = QubitOperator.from_matrix(A).to_pauli()

    def U(qf):
        H.trotterization()(qf,t=-np.pi,steps=5)

    b = np.random.randint(0, 2, size=2**n)

    print("Hermitian matrix A:")
    print(A)

    print("Eigenvalues:")
    print(np.linalg.eigvals(A))

    print("b:")
    print(b)


::

    @terminal_sampling
    def main():

        x = HHL(tuple(b), U, n, 4)
        return x

    res_dict = main()

    for k, v in res_dict.items():
        res_dict[k] = v**0.5

    np.array([res_dict[key] for key in sorted(res_dict)])


Let's compare to the classical solution:

::

    x = (np.linalg.inv(A)@b)/np.linalg.norm(np.linalg.inv(A)@b)
    print(x)


Yup, close enough... That's all folks!

Step-by-step recap
------------------

Let's rewind for a second, take a deep breath, and go through the steps and concepts you learned so far.

Equipped with a theoretical introduction to HHL and outlining the steps required to perform this algorithm, you got to see how to first encode the first 4 steps and making use of the repeat until success feature of Jasp.
Then, putting everything together, we combined the previously defined building blocks (read: Python functions) - the HHL_encoding and QPE - into a simple function. With a brief feature apperance of Hamiltonian simulation you then successfully managed to solve two systems of linear equations.

In conclusion, let’s take a moment to appreciate one last time how elegantly we can call the HHL algorithm:

::

    x = HHL(b, hamiltonian_evolution, n, precision)


As qrispy as always!