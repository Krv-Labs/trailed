.. _index:

=======
TRAILED
=======

**Differentiable Euler Characteristic Transform for Python**

TRAILED provides fast, differentiable ECT (Euler Characteristic Transform) computation with seamless NumPy, pandas, sklearn, and PyTorch integration. Use topological descriptors for machine learning, shape analysis, and data exploration.

Quick Links
-----------

.. grid:: 1 2 3 3
   :gutter: 3
   :padding: 2 2 0 0

   .. grid-item-card:: :octicon:`rocket` Quickstart
      :link: quickstart
      :link-type: ref
      :class-card: intro-card
      :shadow: md

      Compute ECT descriptors from NumPy or pandas in seconds.

   .. grid-item-card:: :octicon:`book` User Guide
      :link: user_guide
      :link-type: ref
      :class-card: intro-card
      :shadow: md

      Installation, configuration, and advanced workflows.

   .. grid-item-card:: :octicon:`code` API Reference
      :link: api-reference
      :link-type: ref
      :class-card: intro-card
      :shadow: md

      Full API documentation for DECT exports.

What is TRAILED?
----------------

The **Euler Characteristic Transform (ECT)** is a topological descriptor that captures shape information through directional filtrations. TRAILED makes ECT computation accessible and efficient for Python workflows:

- **Fast computation** from NumPy arrays and pandas DataFrames
- **Differentiable** for use in gradient-based optimization and neural networks
- **sklearn-compatible** transformers for ML pipelines
- **PyTorch integration** for deep learning workflows

Typical Workflow
----------------

.. mermaid::

   graph LR
      subgraph Input
         A[Point cloud / DataFrame]
      end

      subgraph "Core"
         B["Direction sampling"]
         C["Filtration"]
         D["ECT computation"]
      end

      subgraph "Output"
         E["Descriptor vector"]
         F["Distance matrix"]
      end

      subgraph "Integration"
         G["sklearn pipeline"]
         H["PyTorch model"]
      end

      A --> B
      B --> C
      C --> D
      D --> E
      E --> F
      E --> G
      E --> H

      style Input fill:#f9f9f9,stroke:#999
      style B fill:#D9EDF7,stroke:#31708F,stroke-width:2px
      style D fill:#D9EDF7,stroke:#31708F,stroke-width:2px
      style E fill:#DFF0D8,stroke:#3C763D,stroke-width:2px

**NumPy / pandas**

.. code-block:: python

   import numpy as np
   from trailed import compute_ect_from_numpy

   points = np.random.randn(100, 3)
   descriptor = compute_ect_from_numpy(points, num_thetas=32, resolution=64)

**sklearn Pipeline**

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.svm import SVC
   from trailed.sklearn import ECTTransformer

   pipe = Pipeline([
       ("ect", ECTTransformer(num_thetas=32)),
       ("clf", SVC()),
   ])

**PyTorch**

.. code-block:: python

   import torch
   from trailed.torch import DifferentiableECT

   ect_layer = DifferentiableECT(num_thetas=32, resolution=64)
   descriptor = ect_layer(point_cloud_tensor)

Key Features
------------

**Fast ECT Computation**
   Optimized algorithms for computing Euler characteristic curves across multiple directions.

**Direction Sampling**
   Configurable strategies for sampling directions on the unit sphere (uniform, stratified, custom).

**Resolution Control**
   Adjustable filtration resolution for trading off detail vs. computation time.

**Framework Integration**
   First-class support for NumPy, pandas, sklearn, and PyTorch workflows.

**Differentiable**
   Full gradient support for end-to-end learning with topological features.

Installation
------------

.. code-block:: bash

   pip install trailed

With optional dependencies:

.. code-block:: bash

   pip install trailed[sklearn]     # sklearn integration
   pip install trailed[torch]       # PyTorch integration
   pip install trailed[all]         # everything

For development:

.. code-block:: bash

   git clone https://github.com/Krv-Analytics/trailed.git
   cd trailed
   uv sync --extra dev --extra docs

Supports Python 3.10, 3.11, 3.12.

Next Steps
----------

- :ref:`Quickstart <quickstart>` - Compute your first ECT descriptor
- :ref:`User Guide <user_guide>` - Installation and configuration
- :ref:`API Reference <api-reference>` - Full class and function docs
- :ref:`Integrations <integrations>` - sklearn and PyTorch adapters

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api

.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   overview
   user_guide
   configuration
   integrations

References
----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
