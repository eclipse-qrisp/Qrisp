.. _injection:

Quantum Function Injection
==========================

If you managed to read this far, you realized already that every quantum gate operates in-place on it's qubits. While in-place functions are an established paradigm in classical computing, it would be tempting to just forbid out-of-place functions within Qrisp. This would however require all variable declarations to be executed before anything happens and thus inevitably result in unreadable and cluttered code. 

Out-of-place functions also have their place in quantum and the **injection operator** manages to bridge the two worlds. In particular this structure allows the *programmer* to write/test/maintain quantum functions with out-of-place syntax but the *user* to use them as in-place functions, when required.

The injection operator is called via the ``<<`` symbol and transforms function given as the second operand into an in-place function operating on the first operand:

::

    from qrisp import mcx, QuantumBool

    # Create a simple out-of-place function, computing the
    # AND value of two inputs
    def AND(a, b):
        res = QuantumBool()
        mcx([a, b], res)
        return res

    injection_target = QuantumBool()
    
    # Perform some computation on the injection target
    injection_target.flip()

    # We can now inject the target into the function

    injected_function = (injection_target << AND)

    # Calling this function will now operate on injection_target instead of allocating a new QuantumBool

    a = QuantumBool()
    b = QuantumBool()

    injected_function(a, b)

    print(injection_target.qs)
    # QuantumCircuit:
    # ---------------
    #                     ┌───┐┌───┐
    # injection_target.0: ┤ X ├┤ X ├
    #                     └───┘└─┬─┘
    #                a.0: ───────■──
    #                            │  
    #                b.0: ───────■──
                                  
    # Live QuantumVariables:
    # ----------------------
    # QuantumBool injection_target
    # QuantumBool a
    # QuantumBool b
    

Another example - the injection operator can be used for manual uncomputation of out-of-place arithmetic.

::

    from qrisp import QuantumFloat, z, invert
    
    a = QuantumFloat(5)
    b = QuantumFloat(5)
    a[:] = 3
    b[:] = 4
    
    # The multiplication operator is an out-of-place function
    c = a*b
    
    print(c)
    # Yields: {12: 1.0}
    
    # In an actual algorithm, something would be done now.
    # After that, we can do manual uncomputation by injecting 
    # the multiplication function into the result and 
    # inverting the whole process.
    
    mult_func = lambda x, y : x*y
    
    with invert():
        (c << mult_func)(a, b)
    
    print(c)
    # Yields: {0: 1.0}
    
    # c can now be deallocated.
    c.delete()
    
.. warning::

    The operator can only be used to inject ``QuantumVariables`` of matching size - injecting a size 10 ``QuantumFloat`` into a ``QuantumBool`` will raise an error. When called within Jasp, the operator is restricted to quantum functions, where the return value has never been sliced.