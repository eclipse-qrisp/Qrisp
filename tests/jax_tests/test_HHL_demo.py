"""
********************************************************************************
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
********************************************************************************
"""

def test_HHL_demo():

    import jax
    import qrisp

    qv = qrisp.QuantumVariable(5)

    ############################################################

    # Apply gates to the QuantumVariable.
    qrisp.h(qv[0])
    qrisp.z(qv)  # Z gate applied to all qubits
    qrisp.cx(qv[0], qv[3])

    # Print the quantum circuit.
    print(qv.qs)

    ############################################################

    a = qrisp.QuantumFloat(msize=3, exponent=-2, signed=False)
    qrisp.h(a)
    print(a)

    assert a.get_measurement() == {0.0: 0.125, 0.25: 0.125, 0.5: 0.125, 0.75: 0.125, 1.0: 0.125, 1.25: 0.125, 1.5: 0.125, 1.75: 0.125}

    ############################################################

    a = qrisp.QuantumFloat(3)
    b = qrisp.QuantumFloat(3)

    a[:] = 5
    b[:] = 2

    c = a + b
    d = a - c
    e = d * b
    f = a / b

    print(c)
    print(d)
    print(e)
    print(f)

    assert c.get_measurement() == {7: 1.0}
    assert d.get_measurement() == {-2: 1.0}
    assert e.get_measurement() == {-4: 1.0}
    assert f.get_measurement() == {2.5: 1.0}

    ############################################################

    qb = qrisp.QuantumBool()
    qrisp.h(qb)
    print(qb)
    assert qb.get_measurement() == {False: 0.5, True: 0.5}

    ############################################################

    qb_1 = qrisp.QuantumBool()
    print(qb_1)
    print(qb | qb_1)
    print(qb & qb_1)   

    ############################################################ 

    a = qrisp.QuantumFloat(4)
    qrisp.h(a[3])
    qb_3 = a >= 4
    print(a)
    print(qb_3)
    print(qb_3.qs.statevector())

    assert a.get_measurement() == {0: 0.5, 8: 0.5}
    assert qb_3.get_measurement() == {False: 0.5, True: 0.5}

    ############################################################

    b = qrisp.QuantumFloat(3)
    b[:] = 4
    comparison = a < b
    print(comparison.qs.statevector())

    ############################################################

    def QPE(psi, U, precision=None, res=None):

        if res is None:
            res = qrisp.QuantumFloat(precision, -precision)

        qrisp.h(res)

        # Performs a loop with a dynamic bound in Jasp mode.
        for i in qrisp.jrange(res.size):
            with qrisp.control(res[i]):
                for j in qrisp.jrange(2**i):
                    U(psi)

        return qrisp.QFT(res, inv=True)
    
    ############################################################

    import numpy as np

    def U(psi):
        phi_1 = 0.5
        phi_2 = 0.125

        qrisp.p(phi_1 * 2 * np.pi, psi[0])  # p applies a phase gate
        qrisp.p(phi_2 * 2 * np.pi, psi[1])

    psi = qrisp.QuantumFloat(2)
    qrisp.h(psi)

    res = QPE(psi, U, precision=3)

    print(qrisp.multi_measurement([psi, res]))

    assert qrisp.multi_measurement([psi, res]) == {(0, 0.0): 0.25, (1, 0.5): 0.25, (2, 0.125): 0.25, (3, 0.625): 0.25}

    ############################################################

    @qrisp.terminal_sampling
    def main():
        qf = qrisp.QuantumFloat(2)
        qf[:] = 3

        res = QPE(qf, U, precision=3)

        return res

    main()

    assert main() == {0.625: 1.0}

    ############################################################

    def fake_inversion(qf, res=None):

        if res is None:
            res = qrisp.QuantumFloat(qf.size + 1)

        for i in qrisp.jrange(qf.size):
            qrisp.cx(qf[i], res[qf.size - i])

        return res
    
    ############################################################

    qf = qrisp.QuantumFloat(3, -3)
    qrisp.x(qf[2])
    qrisp.dicke_state(qf, 1)
    res = fake_inversion(qf)
    print(qrisp.multi_measurement([qf, res]))

    assert qrisp.multi_measurement([qf, res]) == {(0.125, 8): 0.3333333333333333, (0.25, 4): 0.3333333333333333, (0.5, 2): 0.3333333333333333}

    ############################################################

    @qrisp.RUS(static_argnums=[0, 1])
    def HHL_encoding(b, hamiltonian_evolution, n, precision):

        # Prepare the state |b>. Step 1
        qf = qrisp.QuantumFloat(n)
        # Reverse the endianness for compatibility with Hamiltonian simulation.
        qrisp.prepare(qf, b, reversed=True)

        qpe_res = QPE(qf, hamiltonian_evolution, precision=precision)  # Step 2
        inv_res = fake_inversion(qpe_res)  # Step 3

        case_indicator = qrisp.QuantumFloat(inv_res.size)

        with qrisp.conjugate(qrisp.h)(case_indicator):
            qbl = (case_indicator >= inv_res)

        cancellation_bool = (qrisp.measure(case_indicator) == 0) & (qrisp.measure(qbl) == 0)

        # The first return value is a boolean value. Additional return values are QuantumVariables.
        return cancellation_bool, qf, qpe_res, inv_res
    
    ############################################################

    def HHL(b, hamiltonian_evolution, n, precision):

        qf, qpe_res, inv_res = HHL_encoding(b, hamiltonian_evolution, n, precision)

        # Uncompute qpe_res and inv_res
        with qrisp.invert():
            QPE(qf, hamiltonian_evolution, res=qpe_res)
            fake_inversion(qpe_res, res=inv_res)

        # Reverse the endianness for compatibility with Hamiltonian simulation.
        for i in qrisp.jrange(qf.size // 2):
            qrisp.swap(qf[i], qf[n - i - 1])

        return qf
    
    ############################################################

    from qrisp.operators import QubitOperator
    import numpy as np

    A = np.array([[3 / 8, 1 / 8], [1 / 8, 3 / 8]])
    b = np.array([1, 1])

    H = QubitOperator.from_matrix(A).to_pauli()

    # By default e^{-itH} is performed. Therefore, we set t=-pi.
    def U(qf):
        H.trotterization()(qf, t=-np.pi, steps=1)    

    ############################################################

    @qrisp.terminal_sampling
    def main():
        x = HHL(tuple(b), U, 1, 3)
        return x

    res_dict = main()

    for k, v in res_dict.items():
        res_dict[k] = v**0.5

    x_ = np.array([res_dict[key] for key in sorted(res_dict)]) 
    print(x_)

    ############################################################

    x = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)
    print(x)

    assert np.linalg.norm(np.abs(x_)-np.abs(x)) < 1e-3

    ############################################################

    def hermitian_matrix_with_power_of_2_eigenvalues(n):
        # Generate eigenvalues as inverse powers of 2.
        eigenvalues = 1 / np.exp2(np.random.randint(1, 4, size=n))

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
        H.trotterization()(qf, t=-np.pi, steps=5)

    b = np.random.randint(0, 2, size=2**n)

    print("Hermitian matrix A:")
    print(A)

    print("Eigenvalues:")
    print(np.linalg.eigvals(A))

    print("b:")
    print(b)

    ############################################################

    @qrisp.terminal_sampling
    def main():
        x = HHL(tuple(b), U, n, 4)
        return x

    res_dict = main()

    for k, v in res_dict.items():
        res_dict[k] = v**0.5

    x_ = np.array([res_dict[key] for key in sorted(res_dict)])  
    print(x_)

    ############################################################

    x = (np.linalg.inv(A) @ b) / np.linalg.norm(np.linalg.inv(A) @ b)
    print(x)

    assert np.linalg.norm(np.abs(x_)-np.abs(x)) < 1e-1

    ############################################################

    def main():
        x = HHL(tuple(b), U, n, 4)
        # Note that we have to return a classical value
        # (in this case the measurement result of the
        # quantum variable returned by the HHL algorithm)
        # Within the above examples, we used the terminal_sampling
        # decorator, which is a convenience feature and allows
        # a much faster sampling procedure.
        # The terminal_sampling decorator expects a function returning
        # quantum variables, while most other evaluation modes require
        # classical return values.
        return qrisp.measure(x)
    
    try:
        import catalyst
    except ImportError:
        return

    jaspr = qrisp.make_jaspr(main)()
    qir_str = jaspr.to_qir()
    # Print only the first few lines - the whole string is very long.
    print(qir_str[:200])


    ############################################################

    A = np.array([[3 / 8, 1 / 8], [1 / 8, 3 / 8]])

    b = np.array([1, 1])

    H = QubitOperator.from_matrix(A).to_pauli()

    # By default e^{-itH} is performed. Therefore, we set t=-pi.
    def U(qf):
        H.trotterization()(qf, t=-np.pi, steps=1)

    @qrisp.qjit
    def main():
        x = HHL(tuple(b), U, 1, 3)

        return qrisp.measure(x)

    samples = []
    for i in range(1):
        samples.append(float(main()))

    print(samples)