.. _programmatic:

============
Programmatic
============

Core package-level exports for script use:

- samplers from ``trailed.dect.sampling``
- dataframe helpers from ``trailed.dect.tabular``
- optional sklearn/torch adapters under ``trailed.dect.plugins.*``

.. code-block:: python

   from trailed import compute_ect_from_numpy
   from trailed import generate_uniform_directions

   directions = generate_uniform_directions(3, method="rand", seed=1)
