from qrisp import h, acc_IQAE, cx, z,  auto_uncompute, QuantumBool


def uniform(*args):
    for arg in args:
        h(arg)


def QMCI(qargs, function, distribution=None):
    """
    Function to facilitate Quantum Monte Carlo Integration

    Parameters
    ----------
    qargs : List(QuantumFloat)
        The QuantumVariable to operate on.
    function : function
        A Python function which takes QuantumFloats as inputs and applies the mathematical function which is to be integrated.
    function : function
        A Python function which takes QuantumFloats as inputs and applies the distribution over which to integrate.

    """
    if distribution==None:
        distribution = uniform

    dupl_args = [arg.duplicate() for arg in qargs]
    dupl_res_qf = function(*dupl_args)
    qargs.append(dupl_res_qf.duplicate())

    for arg in dupl_args:
        arg.delete()
    dupl_res_qf.delete()

    V0=1
    for arg in qargs:
        V0 *= 2**(arg.size+arg.exponent)
    qargs.append(QuantumBool())

    @auto_uncompute
    def state_function(*args):
        qf_x = args[0]
        qf_y = args[1]
        tar = args[2]

        distribution(qf_x)
        h(qf_y)
        qbl = (qf_y < function(qf_x))
        cx(qbl,tar)

    def oracle_function(*args):  
        tar = args[2]
        z(tar)

    a = acc_IQAE(qargs, state_function, oracle_function, eps= 0.01, alpha= 0.01)   

    V = V0*a
    return V