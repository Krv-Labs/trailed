.. _quickstart:

==========
Quickstart
==========

This guide gets you computing topological descriptors in under 5 minutes.

Prerequisites
-------------

- Python 3.10+
- TRAILED installed (``uv pip install trailed`` or ``pip install trailed``)

What is ECT?
------------

The **Euler Characteristic Transform (ECT)** is a topological descriptor that captures shape information. It works by:

1. Projecting points onto multiple directions
2. Computing the Euler characteristic at different filtration levels
3. Returning a vector that uniquely characterizes the shape

For EHR applications, ECT descriptors can capture higher-order structure in patient embedding spaces — detecting patterns that coordinate-based metrics miss.

Basic Usage with NumPy
----------------------

**Step 1: Prepare your data**

TRAILED operates on point clouds or embeddings. For EHR data, this is typically patient embeddings from a representation learning model.

.. code-block:: python

   import numpy as np

   # Example: patient embeddings (100 patients, 64-dimensional)
   patient_embeddings = np.random.randn(100, 64)
   print(f"Embedding shape: {patient_embeddings.shape}")

**Step 2: Compute the ECT descriptor**

.. code-block:: python

   from trailed import compute_ect_from_numpy

   descriptor = compute_ect_from_numpy(
       patient_embeddings,
       num_thetas=32,    # Number of directions
       resolution=64,    # Filtration resolution
   )
   print(f"Descriptor shape: {descriptor.shape}")

The result is a 1D vector capturing the topological structure of your patient cohort.

Using pandas DataFrames
-----------------------

TRAILED works directly with pandas:

.. code-block:: python

   import pandas as pd
   from trailed import compute_ect_from_pandas

   # Patient features as DataFrame
   df = pd.DataFrame(np.random.randn(100, 10), columns=[f"feat_{i}" for i in range(10)])

   descriptor = compute_ect_from_pandas(df, num_thetas=32, resolution=64)

Comparing Cohorts
-----------------

A key use case is comparing topological structure between cohorts — for example, real vs. synthetic patient data:

.. code-block:: python

   from trailed import compute_ect_from_numpy
   import numpy as np

   # Real patient embeddings
   real_embeddings = np.load("real_embeddings.npy")

   # Synthetic patient embeddings
   synthetic_embeddings = np.load("synthetic_embeddings.npy")

   # Compute topological descriptors
   real_ect = compute_ect_from_numpy(real_embeddings, num_thetas=32, resolution=64)
   synthetic_ect = compute_ect_from_numpy(synthetic_embeddings, num_thetas=32, resolution=64)

   # Topological distance
   distance = np.linalg.norm(real_ect - synthetic_ect)
   print(f"Topological distance: {distance:.4f}")

sklearn Integration
-------------------

Use ECT features in machine learning pipelines:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.svm import SVC
   from trailed.plugins.sklearn import EctTransformer

   pipe = Pipeline([
       ("ect", EctTransformer(num_thetas=32, resolution=64)),
       ("clf", SVC()),
   ])

   # X is a list of point clouds, y is labels
   pipe.fit(X_train, y_train)
   accuracy = pipe.score(X_test, y_test)

.. note::

   sklearn integration requires ``uv pip install trailed[sklearn]``.

PyTorch Integration
-------------------

For differentiable ECT in PyTorch deep learning workflows, use the upstream `aidos-lab/dect <https://github.com/aidos-lab/DECT>`_ package:

.. code-block:: bash

   pip install dect @ git+https://github.com/aidos-lab/DECT/

.. code-block:: python

   import torch
   from dect.nn import ECTLayer, ECTConfig

   ect_layer = ECTLayer(ECTConfig(num_thetas=32, resolution=64))

   # Input: batch of point clouds [B, N, D]
   point_clouds = torch.randn(8, 100, 64, requires_grad=True)
   descriptors = ect_layer(point_clouds)

   # Gradients flow through
   loss = descriptors.sum()
   loss.backward()

**Example: Topological Regularization**

.. code-block:: python

   # In a generative model training loop
   real_batch = ...       # Real patient embeddings
   generated_batch = ...  # Generated patient embeddings

   real_ect = ect_layer(real_batch)
   generated_ect = ect_layer(generated_batch)

   # Topological loss encourages structural similarity
   topo_loss = torch.nn.functional.mse_loss(generated_ect, real_ect)
   total_loss = reconstruction_loss + lambda_topo * topo_loss

Tuning Parameters
-----------------

Two key parameters control the descriptor:

- **num_thetas**: Number of directions to sample. More directions = finer angular resolution. Typical values: 16-64.
- **resolution**: Filtration granularity. Higher = more detail. Typical values: 32-128.

.. code-block:: python

   # High resolution for detailed analysis
   desc_detailed = compute_ect_from_numpy(embeddings, num_thetas=64, resolution=128)

   # Low resolution for fast computation
   desc_fast = compute_ect_from_numpy(embeddings, num_thetas=16, resolution=32)

Next Steps
----------

- :doc:`programmatic` - Advanced API usage
- :doc:`intermediate` - Direction sampling strategies
- :doc:`advanced` - Custom filtrations and optimization
- :ref:`Integrations <integrations>` - sklearn and tabular integration details
