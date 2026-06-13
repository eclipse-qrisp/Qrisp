.. _DevGuideWriteaTutorial:

Write a tutorial
================

This page serves as an introductory guide on writing a tutorial or example.

Our tutorials
-------------

You surely came across them already. All of our tutorials live on our :ref:`tutorials section <tutorial>`. 
We are happy to provide extensive overviews on everything that is possible in Qrisp, everything that we have done and many things we think you as the contributor or researcher can work on. 
Any contribution deserves its explanation on how it amazingly solves an important problem or improves the existing approaches, and how others can build on your solution.

So let us talk about writing a tutorial!

Reach Out!
----------

First of all - don't hesitate to talk to us! Do you have any questions? Is anything unclear? Do you want to get started but don't know how? **Reach out to us via our Discord!**

Tutorial or example
-------------------

First of all, let us look at two different types of options to provide insight and an overview of your implementation. 

**Tutorials** live in the aforementioned :ref:`tutorials section <tutorial>`. Each tutorial is a self-contained introduction to the problem and how to solve it. 
Write a *tutorial* if your implementation

* is extensive, 
* provides a full end-to-end workflow, 
* or provides the basis of how a whole quantum computing paradigm can be approached.


**Examples** live in the :ref:`example page (link to Shor's algo example) <ShorExample>`. They are shorter in nature, and truly focus on the code - what purpose it serves and how to use it. 
Write an *example* if your implementation

* is user-friendly, 
* focuses on one (or a few) subtasks 
* and allows for easy embedding into other full workflows.

Write a tutorial
----------------

Each tutorial is a self-contained overview of: 

**The problem formulation**
  A description of where the problem originates from and why it is important to solve it, i.e. use cases.

**The mathematical foundation behind it**
  An explanation of the maths and/or physics behind it. Keep it short and concise. You will need to find a balance between providing sufficient background explanation and keeping the tutorial an exciting read.

**How it is solved using quantum computing**
  What does the research say on how we solve this problem? This is the foundation to give the explanation of your implementation footing. 

**How you implemented this solution in Qrisp**
  A detailed breakdown of how the theoretical solution proposed by researchers is implemented in practice in Qrisp. 

All of these aspects will be required for your tutorial. Our existing tutorials should give you an idea of how to structure yours. 

**Important**:

* We put heavy emphasis on didactical value. Each step should build on the previous. 
* Your tutorial should leave the reader thinking: "*I learned how to translate these concepts to solve the problem into Qrisp code!*" or "*In parallel to learning about these concepts I learned how to use the Qrisp code that solves the problem and can now tackle it myself!*" 
* Keep it short and concise, especially when it comes to math. While mathematical accuracy is essential, dense derivations can distract from the code. Give the reader exactly the math they need to understand the Qrisp implementation, and use references for the rest.
* Our tutorials are ``.ipynb`` (Jupyter Notebook Python) files, so please format your tutorial as such! And make sure that it can be executed as is.

Don't be afraid of getting started! We are happy to help, so please reach out! 


Write an example
----------------

Each example gives the reader an idea of: 

* What user-facing code you implemented
* What it is used for 
* How to actually use it
* Possible limitations

After reading your example the user should be equipped to employ your implementation in their workflow with ease, resulting in a quality-of-life improvement for their development needs! Our examples are generally ``.rst`` (reStructuredText) files, so please stick to the same formatting.
