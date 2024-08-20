
import numpy as np
import math
from typing import Tuple, List, Union, Optional

from qrisp import QuantumFloat
from qrisp import ry, cx

def exponential_distribution(N, alpha):
    """
    Loads the exponential distribution into a QuantumFloat

    Paramters:
    ----------
    N: Int
        The number of qubits for the QuantumFloat. 
    alpha: Float
        The parameter of the distribution

    Returns:
    --------
    qf: QuantumFloat
        the QuantumFloat which has the distribution encoded in its amplitude.

    """

    tau=np.exp(alpha/2**N)

    theta=[]
    for k in range(N):
        theta.append(2*(np.pi/2-np.arctan(np.sqrt(tau**(2**k)))))

    qf = QuantumFloat(N,-N)
    for k in range(N):
        ry(theta[k],qf[k])

    return qf


def gamma_distribution(N,N_qfs,alpha):

    """
    Loads the Gamma distribution as a sum of exponentially distributed QuantumFloats into a QuantumFloat.

    Paramters:
    ----------
    N: Int
        The number of qubits for the QuantumFloat. 
    N_qf: Int
        The number of exponential distributions to sum over 
    alpha: Float
        The parameter of the distribution

    Returns:
    --------
    qf: QuantumFloat
        the QuantumFloat which has the distribution encoded in its amplitude.

    """

    qf = exponential_distribution(N, alpha)
    for index in range(N_qfs-1):
        qf2 = exponential_distribution(N, alpha)
        qf += qf2
        qf2.uncompute()

    return qf



def laplace_distribution(N,alpha1, alpha2):

    """
    Loads the LaPlace distribution into a QuantumFloat.

    Paramters:
    ----------
    N: Int
        The number of qubits for the QuantumFloat. 
    alpha1: Float
        The first parameter of the distribution.
    alpha2: Float
        The second parameter of the distribution.

    Returns:
    --------
    qf: QuantumFloat
        the QuantumFloat which has the distribution encoded in its amplitude.

    """
        
    tau1=np.exp(alpha1/2**(N-1))
    tau2=np.exp(alpha2/2**(N-1))

    theta1=[]
    for k in range(N):
        theta1.append(np.pi+2*(np.pi/2-np.arctan(np.sqrt(tau1**(2**k)))))
    theta1.append(np.pi/2)

    theta2=[]
    for k in range(N):
        theta2.append(np.pi+2*(np.pi/2-np.arctan(np.sqrt(tau2**(2**k)))))
    theta2.append(np.pi/2)

        
    qf1 = QuantumFloat(N,-N,signed=True)
    for k in range(N+1):
        ry(np.pi+theta1[k],qf1[k])

    for k in range(N):
        cx(qf1[N],qf1[N-k-1])

        
    qf2 = QuantumFloat(N,-N,signed=True)
    for k in range(N+1):
        ry(np.pi+theta2[k],qf2[k])

    for k in range(N):
        cx(qf2[N],qf2[N-k-1])

    return qf1 +qf2




def negbinom_distribution(N,alpha):
    """
    Loads an approximation of the negative binomial distribution into a QuantumFloat.

    Paramters:
    ----------
    N: Int
        The number of qubits for the QuantumFloat. 
    alpha1: Float
        The first parameter of the distribution.
    alpha2: Float
        The second parameter of the distribution.

    Returns:
    --------
    qf: QuantumFloat
        the QuantumFloat which has the distribution encoded in its amplitude.

    """
    tau=np.exp(alpha/2**N)
    print(tau)

    theta=[]
    for k in range(N):
        theta.append(2*(np.pi/2-np.arctan(np.sqrt(tau**(2**k)))))
        
    qf1 = QuantumFloat(N,-N)
    for k in range(N):
        ry(theta[k],qf1[k])
        
    qf2 = QuantumFloat(N,-N)
    for k in range(N):
        ry(theta[k],qf2[k])
        
    qf3 = QuantumFloat(N,-N)
    for k in range(N):
        ry(theta[k],qf3[k])
    
    qf1 += qf2+qf3
    qf4 = qf1+qf2
    qf1.uncompute()
    qf2.uncompute()
    qf3.uncompute()

    return qf4





def normal_distribution(
    #num_qubits: Union[int, List[int]],
    num_values: Union[int, List[int]],
    
    precision: Optional[Union[float, List[float]]] = None,
    mu: Optional[Union[float, List[float]]] = 0,
    sigma: Optional[Union[float, List[float]]] = 1,
    signed: bool =True
    
    ):

    """
    Loads a normal distribution into a QuantumFloat. This is a classically preprocessed amplitude encoding. The encoding requires you to make informed decisions on the inputs.
    

    Paramters:
    ----------
    num_qubits: Int
        The number of qubits for the QuantumFloat. 
    num_values: Int
        The number of values from the distribution
    signed: bool
        Boolean to indicate whether the QuantumFloat is signed
    precision: Int
        The precision of the QuantumFloat
    mu: Int
        Mean of the distribution
    sigma: Float
        Sigma of the distribution

    Returns:
    --------
    qf: QuantumFloat
        the QuantumFloat which has the distribution encoded in its amplitude.


    """
    
    # set default arguments
    #dim = 1 if isinstance(num_qubits, int) else len(num_qubits)
    dim = 1

    #if bounds is None:
        #bounds = (0, 1) if dim == 1 else [(0, 1)] * dim
    
    #num_values = 2**num_qubits
    num_values = num_values
    
    num_qubits = math.ceil(math.log2(num_values))
    #print(num_qubits)
    precision_float = pow(2,precision)
    
    qf = QuantumFloat(num_qubits,precision, signed=signed)
    #print(len(qf))

    # x should be what ? my mu rounded to supported precision
    # and then -+ the number of supported entries on
    def myround(num, base=precision_float):
        return base * round(num/base)
    
    mu_rounded = myround(mu)
    x=[]

    for i in reversed(range(1, int(num_values) +1) ):
        x.append(mu_rounded - i *precision_float)
    for i in range(int(num_values)):
        x.append(mu_rounded + i *precision_float)
    probabilities = []

    from scipy.stats import multivariate_normal

    for x_i in x:
        # treat negative values??
        probability = multivariate_normal.pdf(x_i, mu, sigma) 
        probabilities += [probability]
    normalized_probabilities = probabilities / np.sum(probabilities)
    #print(normalized_probabilities)


    dict = {key: value for key, value in zip(x, np.sqrt(normalized_probabilities))}
    #print(dict)
    qf.init_state(dict)

    return qf 
    


def lognormal_distribution(
    num_qubits: Union[int, List[int]],
    num_values: Union[int, List[int]],
    precision: Optional[Union[float, List[float]]] = None,
    mu: Optional[Union[float, List[float]]] = None,
    sigma: Optional[Union[float, List[float]]] = None,
    bounds: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None

    ):
    
    """
    Loads a logNormal distribution into a QuantumFloat. This is a classically preprocessed amplitude encoding. The encoding requires you to make informed decisions on the inputs.
    

    Paramters:
    ----------
    num_qubits: Int
        The number of qubits for the QuantumFloat. 
    num_values: Int
        The number of values from the distribution
    signed: bool
        Boolean to indicate whether the QuantumFloat is signed
    precision: Int
        The precision of the QuantumFloat
    mu: Int
        Mean of the distribution
    sigma: Float
        Sigma of the distribution

    Returns:
    --------
    qf: QuantumFloat
        the QuantumFloat which has the distribution encoded in its amplitude.


    """
    
    

    # set default arguments
    dim = 1 if isinstance(num_qubits, int) else len(num_qubits)
    if mu is None:
        mu = 0 if dim == 1 else [0] * dim

    if sigma is None:
        sigma = 1 if dim == 1 else np.eye(dim)

    if bounds is None:
        bounds = (0, 1) if dim == 1 else [(0, 1)] * dim
    
    #num_values = 2**num_qubits
    num_values = num_values
    precision_float = pow(2,precision)

    qf = QuantumFloat(num_qubits,precision, signed=False)

    def myround(x, base=precision_float):
        return base * round(x/base)
    
    mu_rounded = myround(mu)
    x=[]
    for i in reversed(range(1, int(num_values/2) +1) ):
        x.append(mu_rounded - i *precision_float)
    for i in range(int(num_values/2)):
        x.append(mu_rounded + i *precision_float)

    probabilities = []
    from scipy.stats import multivariate_normal

    for x_i in x:
        # map probabilities from normal to log-normal reference:
        # https://stats.stackexchange.com/questions/214997/multivariate-log-normal-probabiltiy-density-function-pdf
        if np.min(x_i) > 0:
            det = 1 / np.prod(x_i)
            probability = multivariate_normal.pdf(np.log(x_i), mu, sigma) * det
        else:
            probability = 0
        probabilities += [probability]
    normalized_probabilities = probabilities / np.sum(probabilities)
    print(normalized_probabilities)


    dict = {key: value for key, value in zip(x, np.sqrt(normalized_probabilities))}
    print(dict)
    qf.init_state(dict)

    return qf 



def scipy_distribution(  
    distribution,
    num_qubits: Union[int, List[int]],
    num_values: Union[int, List[int]],
    signed: bool =False,
    precision: Optional[Union[float, List[float]]] = None,
    mu: Optional[Union[float, List[float]]] = None,
    sigma: Optional[Union[float, List[float]]] = None,
    bounds: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None
    ):
        

    """
    Loads a SciPy distribution into a QuantumFloat. This is a classically preprocessed amplitude encoding. The encoding requires you to make informed decisions on the inputs.
    

    Paramters:
    ----------
    num_qubits: Int
        The number of qubits for the QuantumFloat. 
    num_values: Int
        The number of values from the distribution
    signed: bool
        Boolean to indicate whether the QuantumFloat is signed
    precision: Int
        The precision of the QuantumFloat
    mu: Int
        Mean of the distribution
    sigma: Float
        Sigma of the distribution

    Returns:
    --------
    qf: QuantumFloat
        the QuantumFloat which has the distribution encoded in its amplitude.


    """
    
    dim = 1 if isinstance(num_qubits, int) else len(num_qubits)
    if mu is None:
        mu = 0 if dim == 1 else [0] * dim
    if sigma is None:
        sigma = 1 if dim == 1 else np.eye(dim)
    if bounds is None:
        bounds = (0, 1) if dim == 1 else [(0, 1)] * dim
    
    #num_values = 2**num_qubits
    num_values = num_values
    precision_float = pow(2,precision)
    qf = QuantumFloat(num_qubits,precision, signed=signed)

    def myround(x, base=precision_float):
        return base * round(x/base)
    mu_rounded = myround(mu)
    x=[]

    for i in reversed(range(1, int(num_values/2) +1) ):
        x.append(mu_rounded - i *precision_float)
    for i in range(int(num_values/2)):
        x.append(mu_rounded + i *precision_float)
    probabilities = []

    pdf = distribution.pdf
    for x_i in x:
        # treat negative values??
        if np.min(x_i) > 0:
            probability = pdf(x_i, mu, sigma) 
        else:
            probability = 0
        probabilities += [probability]
    normalized_probabilities = probabilities / np.sum(probabilities)

    dict = {key: value for key, value in zip(x, np.sqrt(normalized_probabilities))}
    qf.init_state(dict)

    return qf 
