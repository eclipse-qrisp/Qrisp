# Why Quantum Computing?

Welcome to the qrisp learning module! If you are reading this text, you already made a good decision today. Together, we want to explore the world of quantum computing. 
Let's start our journey with this very first chapter and find out why you should spend your time learning about quantum computers. 

## Quantum gate computing vs Quantum annealing

In this documentation, we will focus on gate computing instead of annealing (don't worry, most parts of the foundation stay the same if you want to dip into annealing. For an in-depth explanation, see https://arxiv.org/pdf/2207.01827.pdf). Quantum annealers specialize in optimization problems, finding the best solution for a problem by exploring all possible solutions simultaneously and gradually "tune" them toward the optimal answer, like slowly cooling a system to find its lowest energy state.

The financial sector could heavily profit from this technique, for example from portfolio optimization. Being able to handle vast amounts of data and swiftly identify risks and profits would create a huge advantage.   
Currently, estimated 10 to 40 billion dollars are lost every year to fraud and faulty data management. As present fraud detection is highly inaccurate with a high number of false alarms, quantum technology can revolutionize customer targeting and prediction modeling by efficiently analyzing large volumes of behavioral data to offer personalized financial products and services in real-time.


## time complexity 

If you have dabbled in computer science before, you probably came across this picture: 
![Alt text](https://i.stack.imgur.com/6zHEt.png) 
<span style="color:grey;font-weight:300;font-size:8px"> 
Source: Theprogrammersfirst (https://theprogrammersfirst.wordpress.com/2020/07/22/what-is-a-plain-english-explanation-of-big-o-notation/)
</span>

Explaining the different time complexity classes of algorithms. The algorithms are divided based on their estimated runtime for a specific input size $n$. Many algorithms that we are interested in have at least quadratic complexity and are therefore not viable in everyday life. A big example is RSA cryptography: Most encryption counts on the fact that finding two prime numbers that factor a given number is essentially impossible in a short time. With a big enough quantum computer however, that task (for pretty much any given number) can be solved in a day. The so-called Shor's algorithm exploits quantum phenomena such as superposition and entanglement to achieve this speedup. In the section "necessary physics", we will discover what exactly these words mean. They make quantum advantage possible and are the main reason why we are interested in quantum computers. 

## complexity theory 

Similar to time complexity, we already know complexity classes from classical computing. 
The classes P and NP should seem familiar, even if we cannot confirm whether P = NP or P ≠ NP. Nevertheless, both classes simulate a Turing machine in polynomial time, deterministic (P) or nondeterministic (NP) (you can imagine a Turing machine like a mathematical model of a very basic computer). Furthermore, the class BPP contains all problems that can be solved on a Turing Machine in polynomial time in a probabilistic manner. We cannot guarantee the right solution every time, but we demand at least 2/3 of all answers to be correct (BPP= bounded-error probabilistic polynomial time). Similarly, the complexity class BQP (bounded-error quantum polynomial time) solves a problem on a quantum computer in polynomial time, up to an error of 1/3 (we will talk more about the error rate of quantum computers in the chapter "error handling"). 
Shors's algorithm for factoring is suspected to be in BQP, but not BPP, meaning we can simulate the algorithm in polynomial time on a quantum computer, but (probably) not on a classical computer like a Turing machine. It is assumed that multiple problems lay in BQP, but not BPP, making quantum algorithms attractive for us.

![Alt text](https://www.nicepng.com/png/detail/420-4201382_complexity-png.png)
<span style="color:grey;font-weight:300;font-size:8px"> 
Source: Nicepng (https://www.nicepng.com/png/detail/420-4201382_complexity-png.png)
</span>

As pictured, the suspected relation of the mentioned complexity classes is $P \subseteq BPP \subseteq BQP$

## Today's Quantum Computers

Classical computers work with bits, the smallest unit of information, that can either be 0 or 1. If we combine them, our information will grow linearly. Quantum Computers work on qubits, that can store information like classical qubits, but given their quantum nature, have interesting physical properties: They can take values other than 0 and 1. A quantum state is defined by two complex coordinates $q = (x, y$) on a sphere, satisfying the equation $|x|^2 + |y|^2 = 1$. Here, $x^2$ and $y^2$ are the probabilities that the wavefunction collapses to 1 or 0 when measuring (this is why they need to add up to 1). For example, a valid qubit state is 
$q= (\frac{1}{\sqrt{2}},\frac{1}{\sqrt{2}})$, 
since 
$(\frac{1}{\sqrt{2}})^2 + (\frac{1}{\sqrt{2}})^2 = \frac{1}{\sqrt{2}^2}+\frac{1}{\sqrt{2}^2}= \frac{1}{2} + \frac{1}{2} = 1$.
In this state, the collapse to 0 and 1 are equally probable, we call it an equal superposition. 
When combining qubits together, we now get exponential growth $2^n$ instead of linear growth. Because of superposition, the qubits can be in all possible states $2^n$ at the same time. This means that $n$ qubits are equivalent to $2^{n}$ bits. 

Moore's law states that classical computer chips double in computational power every 1.5 -2 years, leading to exponential growth. Let's confirm that:  

![Alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Moore%27s_Law_Transistor_Count_1970-2020.png/2560px-Moore%27s_Law_Transistor_Count_1970-2020.png)


(Notice that the scale is logarithmic, hence the pictured growth is in fact exponential.) 

Now, a similar trend can be observed in built quantum computers: They double in qubits count in a regular time. But since their computational power is already exponential to classical computers, exponential qubit growth means a double exponential increase in computational power. 
While classical computers are limited by practical dimensions like the size of atoms or the number of available atoms in the universe, quantum computers could have the same (if not more) computational power with less resources.
If this trend continues, quantum computer will soon outpower classical computers by far. 
Problems that grow exponentially on classical computers would only grow linearly on quantum hardware. One example is the simulation of atoms: The chemical reaction of three hydrogen atoms is already challenging, even for a supercomputer. With every added atom, the needed calculations double. Now imagine the really interesting stuff like proteins or cancer cells, absolutely impossible to reliably model today. 
With quantum computing, bioinformatics would be on another level, to just mention one of the many application areas of quantum computing. 

Right now, experimental quantum comuters are limited by the available qubit count and the error rates of the operations.  Multiple labs are invested in building bigger and more accurate quantum computers to actually see the quantum advantage described earlier. In addition, qubits are very delicate, so far quantum computers only exist in labs. Even there, external fields disturb the quantum properties and errors are no rarity. Due to this instability, faultiness and decoherence (quantum behavior is lost to the environment), the amount of required physical qubits build into a system is much higher than the number of theoretical logical qubits thought of in a circuit (this relationship lays anywhere between 10:1 to 1.000:1, depending on the circuit). This also taints our relationship  $|bits| = 2^{|qubits|}$, since this only counts for logical qubits.  
Because this is a physical system, all results we will get after measuring are probabilistic, meaning that we only have a certain probability to get this result. When we repeat the same procedure, even on the same hardware, there is no guarantee to achieve the same result. Therefore, experiments need to be revised many times and our algorithms need to deliver a high probability. 

Even though there are many unsolved problems, don't be intimidated by quantum computing! Reliable hardware is currently being developed and as it is a newly emerging field, you too could contribute to its growth. 
In the next sections, we will walk through necessary math and physics to understand the basics of quantum computing. After that, we will look at simple quantum algorithms and work our way up to more complex ones. 

## Summary 

+ quantum computing allows for more time efficient algorithms through the special mechanics like superposition and entanglement, benefitting various industries 
+ qubits code information as probabilities of 0 and 1, satisfying $|x^2|+ |y^2| = 1$
+ quantum computers have the potential to exceed Moore's law in terms of computational power, since qubits hold exponential information gain
+ building quantum computers is a delicate and time-consuming task, still prone to error and outer influences 
