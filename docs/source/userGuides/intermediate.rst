.. _intermediate:

============
Intermediate
============

Fine-tune direction sampling and descriptor parameters for your use case.

Direction Sampling
------------------

The ``num_thetas`` parameter controls how many directions are sampled on the unit sphere. More directions provide finer angular resolution but increase computation time.

.. code-block:: python

   from trailed import compute_ect_from_numpy

   # Fewer directions: faster, coarser
   desc_fast = compute_ect_from_numpy(data, num_thetas=16, resolution=32)

   # More directions: slower, finer detail
   desc_detailed = compute_ect_from_numpy(data, num_thetas=64, resolution=32)

**Recommendations:**

- 16-32 directions for exploratory analysis
- 32-64 directions for production use
- 64+ directions when distinguishing subtle differences

Resolution Control
------------------

The ``resolution`` parameter controls filtration granularity — how finely the ECT curve is discretized.

.. code-block:: python

   # Lower resolution: smaller descriptors, faster
   desc_small = compute_ect_from_numpy(data, num_thetas=32, resolution=32)

   # Higher resolution: larger descriptors, more detail
   desc_large = compute_ect_from_numpy(data, num_thetas=32, resolution=128)

**Recommendations:**

- Resolution 32-64 for most use cases
- Resolution 128+ for highly detailed analysis
- Match resolution to the complexity of your data

Normalization
-------------

For comparing descriptors across datasets with different scales, consider normalizing:

.. code-block:: python

   import numpy as np

   desc = compute_ect_from_numpy(data, num_thetas=32, resolution=64)

   # L2 normalization
   desc_normalized = desc / np.linalg.norm(desc)

Reproducibility
---------------

For reproducible results, use a fixed random seed when sampling directions:

.. code-block:: python

   import numpy as np

   np.random.seed(42)
   desc = compute_ect_from_numpy(data, num_thetas=32, resolution=64)
