.. _index:

=======
TRAILED
=======

**Topological Regularization and Integrity Learning for EHR Data**

.. warning::

   TRAILED is under active development. The current release provides the foundational ECT (Euler Characteristic Transform) implementation. Healthcare-specific methods — including density-aware descriptors, patient manifold construction, and clinical fidelity metrics — are in progress.

TRAILED provides topological representation learning methods for Electronic Health Record (EHR) data. Built on the differentiable Euler Characteristic Transform (ECT), TRAILED enables topological analysis of patient trajectories, synthetic data validation, and clinical fidelity assessment.

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

What is TRAILED?
----------------

TRAILED addresses two persistent challenges in longitudinal EHR analysis and synthetic data generation:

**Mode Collapse Detection**
   Rare but clinically significant phenotypes are often underrepresented or absent in synthetic data. Standard statistical metrics fail to detect these gaps. TRAILED's topological descriptors capture higher-order structure, revealing coverage failures invisible to coordinate-based methods.

**Pathological Interpolation**
   Synthetic patient trajectories may pass through biologically implausible states — impossible lab value transitions, contradictory comorbidities, or clinically incoherent sequences. TRAILED characterizes the patient manifold to distinguish viable pathways from "No-Go" regions.

Core Capabilities
-----------------

- **Topological Descriptors**: ECT-based representations that capture shape and structure in clinical latent spaces
- **Differentiable**: Full gradient support for end-to-end learning and training-time regularization
- **Patient Manifold Analysis**: Characterize trajectory spaces and identify impossible state transitions
- **Fidelity Metrics**: Quantify real-vs-synthetic alignment in coordinate-free topological space

Architecture
------------

.. mermaid::

   graph LR
      subgraph Input
         A[Patient Embeddings / Trajectories]
      end

      subgraph "Topological Core"
         B["Direction sampling"]
         C["Filtration"]
         D["ECT computation"]
      end

      subgraph "Output"
         E["Topological Descriptor"]
         F["Fidelity Score"]
      end

      subgraph "Applications"
         G["Training Regularizer"]
         H["Quality Assessment"]
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

   # Regularize generative model
   real_ect = ect_layer(real_batch)
   synthetic_ect = ect_layer(generated_batch)
   topo_loss = torch.nn.functional.mse_loss(synthetic_ect, real_ect)

**sklearn Pipeline**

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.svm import SVC
   from trailed.sklearn import ECTTransformer

   pipe = Pipeline([
       ("ect", ECTTransformer(num_thetas=32)),
       ("clf", SVC()),
   ])

Key Features
------------

**Differentiable ECT**
   Optimized implementation supporting forward and backward passes for gradient-based optimization.

**Direction Sampling**
   Configurable strategies for sampling directions — uniform, stratified, or custom direction sets.

**Resolution Control**
   Adjustable filtration resolution for trading off detail vs. computation time.

**Framework Integration**
   First-class support for NumPy, pandas, sklearn, and PyTorch workflows.

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
