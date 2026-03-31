.. _programmatic:

============
Programmatic
============

Core package-level exports for script use:

- samplers from ``trailed.sampling``
- dataframe helpers from ``trailed.tabular``
- sklearn adapters under ``trailed.plugins.sklearn``

For PyTorch neural network use cases, use the upstream `aidos-lab/dect <https://github.com/aidos-lab/DECT>`_ package.

.. code-block:: python

   from trailed import compute_ect_from_numpy
   from trailed import generate_uniform_directions

   directions = generate_uniform_directions(32, 3, seed=42)
