.. _index:

=======
TRAILED
=======

**Topological Regularization and Integrity Learning for EHR Data**

.. warning::

   TRAILED is under active development. The current release provides the foundational ECT (Euler Characteristic Transform) implementation. Healthcare-specific methods — including density-aware descriptors, patient manifold construction, and clinical fidelity metrics — are in progress.

TRAILED is a topological representation learning library for Electronic Health Record (EHR) data. It provides methods for analyzing patient trajectories, validating synthetic data, and assessing clinical fidelity using topological techniques.

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

      Compute topological descriptors from patient data in minutes.

   .. grid-item-card:: :octicon:`book` User Guide
      :link: user_guide
      :link-type: ref
      :class-card: intro-card
      :shadow: md

      Installation, configuration, and clinical workflows.

   .. grid-item-card:: :octicon:`code` API Reference
      :link: api-reference
      :link-type: ref
      :class-card: intro-card
      :shadow: md

      Full API documentation for TRAILED modules.

Why TRAILED?
------------

Longitudinal EHR analysis and synthetic data generation face two persistent challenges that standard metrics fail to detect:

**Mode Collapse**
   Rare but clinically significant phenotypes — pediatric rare diseases, underrepresented demographics — are often absent from synthetic datasets. Models trained on such data fail silently on these populations. Pairwise statistical metrics miss these coverage gaps because they cannot capture higher-order structure.

**Pathological Interpolation**
   Generative models produce synthetic patient trajectories that pass through biologically implausible states: impossible lab value transitions, contradictory comorbidities, or clinically incoherent sequences. These failures create safety risks and degrade downstream model reliability.

TRAILED addresses these problems using **topological methods** that capture global structure in patient trajectory spaces — detecting patterns invisible to coordinate-based metrics.

Core Capabilities
-----------------

- **Topological Descriptors**: Representations that capture shape and structure in clinical latent spaces
- **Differentiable**: Full gradient support for training-time regularization of generative models
- **Patient Manifold Analysis**: Characterize trajectory spaces and identify impossible state transitions
- **Fidelity Metrics**: Quantify real-vs-synthetic alignment in coordinate-free topological space

Architecture
------------

.. mermaid::

   graph LR
      subgraph Input
         A[Patient Embeddings / Trajectories]
      end

      subgraph "TRAILED Core"
         B["Topological Analysis"]
         C["Manifold Construction"]
         D["Fidelity Scoring"]
      end

      subgraph "Applications"
         E["Training Regularizer"]
         F["Synthetic Data QA"]
         G["Trajectory Analysis"]
      end

      A --> B
      A --> C
      B --> D
      C --> D
      D --> E
      D --> F
      D --> G

      style Input fill:#f9f9f9,stroke:#999
      style B fill:#D9EDF7,stroke:#31708F,stroke-width:2px
      style D fill:#DFF0D8,stroke:#3C763D,stroke-width:2px

Example Usage
-------------

**Computing Topological Descriptors**

.. code-block:: python

   import numpy as np
   from trailed import compute_ect_from_numpy

   # Patient embeddings from EHR data
   patient_embeddings = np.load("embeddings.npy")
   descriptor = compute_ect_from_numpy(patient_embeddings, num_thetas=32, resolution=64)

**Training Regularization**

.. code-block:: python

   from trailed.torch import EctLayer, EctConfig

   ect_layer = EctLayer(EctConfig(num_thetas=32, resolution=32))

   # Regularize generative model to preserve topological structure
   real_topo = ect_layer(real_batch)
   synthetic_topo = ect_layer(generated_batch)
   topo_loss = torch.nn.functional.mse_loss(synthetic_topo, real_topo)

**Synthetic Data Fidelity**

.. code-block:: python

   # Compare topological structure of real vs synthetic cohorts
   real_descriptor = compute_ect_from_numpy(real_embeddings, num_thetas=64, resolution=64)
   synthetic_descriptor = compute_ect_from_numpy(synthetic_embeddings, num_thetas=64, resolution=64)

   fidelity_score = np.linalg.norm(real_descriptor - synthetic_descriptor)

Installation
------------

We recommend installing with `uv <https://docs.astral.sh/uv/>`_ for fast, reliable dependency resolution:

.. code-block:: bash

   uv pip install trailed

With optional dependencies:

.. code-block:: bash

   uv pip install trailed[sklearn]     # sklearn integration
   uv pip install trailed[torch]       # PyTorch integration
   uv pip install trailed[all]         # everything

Or with pip:

.. code-block:: bash

   pip install trailed

For development:

.. code-block:: bash

   git clone https://github.com/Krv-Analytics/trailed.git
   cd trailed
   uv sync --extra dev --extra docs

Supports Python 3.10, 3.11, 3.12.

Next Steps
----------

- :ref:`Quickstart <quickstart>` - Compute your first topological descriptor
- :ref:`Overview <overview>` - Technical background and roadmap
- :ref:`User Guide <user_guide>` - Installation and configuration
- :ref:`API Reference <api-reference>` - Full class and function docs

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
