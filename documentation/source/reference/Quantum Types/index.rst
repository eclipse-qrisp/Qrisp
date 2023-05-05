.. _QuantumTypes:

Quantum Types
=============

.. toctree::
   :maxdepth: 2
   :hidden:
   
   QuantumFloat
   QuantumBool
   QuantumChar
   QuantumString

Quantum typing in Qrisp is smoothly integrated into the typing infractructure of Python, implying it can be used to specify the types of function arguments and return values. By using quantum typing, developers can write more self-documenting code and improve the maintainability of their projects.

A quantum type in Qrisp is a subclass of :ref:`QuantumVariable`.

We include three basic built-in quantum types.

* :ref:`QuantumFloat <QuantumFloat>`

* :ref:`QuantumBool <QuantumBool>`

* :ref:`QuantumChar <QuantumChar>`

:ref:`QuantumString` is technically not a quantum type, since it inherits from :ref:`QuantumArray` but we still mention it here.

Creating custom quantum types
-----------------------------

Creating a new quantum type within Qrisp is conveniently achieved by inheriting from :ref:`QuantumVariable` and modifying the :meth:`decoder <qrisp.QuantumVariable.decoder>`. In the following code snippet we demonstrate how to create a date quantum datatype using the `datetime package <https://docs.python.org/3/library/datetime.html>`_.

::

   import datetime
   from qrisp import QuantumVariable

   class QuantumDate(QuantumVariable):
       
       def __init__(self, size, starting_date):
           self.starting_date = starting_date
           QuantumVariable.__init__(self, size)
           
       def decoder(self, i):
           return self.starting_date + datetime.timedelta(i)
       

   today = datetime.date.today()
   tomorrow = today + datetime.timedelta(1)
   even_later = today + datetime.timedelta(4)
   
   #Create a QuantumDate instance
   qd = QuantumDate(size = 3, starting_date = today)

   #Initiate some wave function (amplitudes will be normalized)
   qd[:] = {today : 1j, tomorrow : 0.5, even_later : -0.5}

Evaluate the measurement

>>> print(qd)
{datetime.date(2023, 3, 7): 0.6667, datetime.date(2023, 3, 8): 0.1667, datetime.date(2023, 3, 11): 0.1667}


.. note::
   The results of the decoder have to be hashable.