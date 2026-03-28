.. _quickstart:

==========
Quickstart
==========

This guide gets you computing ECT descriptors in under 5 minutes.

Prerequisites
-------------

- Python 3.10+
- TRAILED installed (``pip install trailed``)

What is ECT?
------------

The **Euler Characteristic Transform (ECT)** is a topological descriptor that captures shape information. It works by:

1. Projecting points onto multiple directions
2. Computing the Euler characteristic at different filtration levels
3. Returning a vector that uniquely characterizes the shape

Basic Usage with NumPy
----------------------

**Step 1: Create a point cloud**

.. code-block:: python

   import numpy as np

   # Generate a random point cloud
   rng = np.random.default_rng(42)
   points = rng.normal(size=(100, 3))
   print(f"Point cloud shape: {points.shape}")

**Step 2: Compute the ECT descriptor**

.. code-block:: python

   from trailed import compute_ect_from_numpy

   descriptor = compute_ect_from_numpy(
       points,
       num_thetas=32,    # Number of directions
       resolution=64,    # Filtration resolution
   )
   print(f"Descriptor shape: {descriptor.shape}")

The result is a 1D vector capturing the topological structure of your point cloud.

Using pandas DataFrames
-----------------------

TRAILED works directly with pandas:

.. code-block:: python

   import pandas as pd
   from trailed import compute_ect_from_pandas

   df = pd.DataFrame({
       "x": np.random.randn(100),
       "y": np.random.randn(100),
       "z": np.random.randn(100),
   })

   descriptor = compute_ect_from_pandas(df, num_thetas=32, resolution=64)

Comparing Shapes
----------------

ECT descriptors enable shape comparison via distance metrics:

.. code-block:: python

   from scipy.spatial.distance import cdist

   # Generate two different point clouds
   cloud1 = rng.normal(loc=0, size=(100, 3))
   cloud2 = rng.normal(loc=2, size=(100, 3))  # Shifted

   desc1 = compute_ect_from_numpy(cloud1, num_thetas=32, resolution=64)
   desc2 = compute_ect_from_numpy(cloud2, num_thetas=32, resolution=64)

   # Compute distance
   distance = np.linalg.norm(desc1 - desc2)
   print(f"ECT distance: {distance:.4f}")

sklearn Integration
-------------------

Use ECT features in machine learning pipelines:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.svm import SVC
   from trailed.sklearn import ECTTransformer

   pipe = Pipeline([
       ("ect", ECTTransformer(num_thetas=32, resolution=64)),
       ("clf", SVC()),
   ])

   # X is a list of point clouds, y is labels
   pipe.fit(X_train, y_train)
   accuracy = pipe.score(X_test, y_test)

.. note::

   sklearn integration requires ``pip install trailed[sklearn]``.

PyTorch Integration
-------------------

For differentiable ECT in deep learning:

.. code-block:: python

   import torch
   from trailed.torch import DifferentiableECT

   ect_layer = DifferentiableECT(num_thetas=32, resolution=64)

   # Input: batch of point clouds [B, N, D]
   point_clouds = torch.randn(8, 100, 3, requires_grad=True)
   descriptors = ect_layer(point_clouds)

   # Gradients flow through!
   loss = descriptors.sum()
   loss.backward()

.. note::

   PyTorch integration requires ``pip install trailed[torch]``.

Tuning Parameters
-----------------

Two key parameters control the descriptor:

- **num_thetas**: Number of directions to sample. More directions = finer angular resolution. Typical values: 16-64.
- **resolution**: Filtration granularity. Higher = more detail. Typical values: 32-128.

.. code-block:: python

   # High resolution for detailed analysis
   desc_detailed = compute_ect_from_numpy(points, num_thetas=64, resolution=128)

   # Low resolution for fast computation
   desc_fast = compute_ect_from_numpy(points, num_thetas=16, resolution=32)

Next Steps
----------

- :doc:`programmatic` - Advanced API usage
- :doc:`intermediate` - Direction sampling strategies
- :doc:`advanced` - Custom filtrations and optimization
- :ref:`Integrations <integrations>` - sklearn and PyTorch details
