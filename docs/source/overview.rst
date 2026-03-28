.. _overview:

========
Overview
========

TRAILED provides **topological representation learning methods for EHR data**, built on the differentiable Euler Characteristic Transform (ECT). It bridges topological data analysis with clinical machine learning.

.. note::

   TRAILED is under active development. The current release provides the foundational ECT implementation. Healthcare-specific extensions are in progress.

The Problem
-----------

Longitudinal EHR data and synthetic data generation face two persistent challenges:

**Mode Collapse**
   Rare but clinically significant phenotypes — pediatric rare diseases, underrepresented demographics — are often absent from synthetic datasets. Models trained on such data fail on these populations. Standard statistical metrics cannot detect these coverage gaps because they rely on pairwise comparisons that miss higher-order structure.

**Pathological Interpolation**
   Generative models produce synthetic patient trajectories that pass through biologically implausible states: impossible lab value transitions, contradictory comorbidity sequences, or clinically incoherent progressions. These failures degrade downstream model reliability and create safety risks.

Why Topology?
-------------

Traditional fidelity metrics are **coordinate-dependent** and **local** — they compare distributions point-by-point but cannot capture the global structure of patient trajectory spaces. Topological methods provide:

**Coordinate-Free Representations**
   ECT descriptors encode shape without relying on specific coordinate systems, making them robust to embedding choices.

**Higher-Order Structure**
   Topology captures connectivity, holes, and voids — the "shape" of data distributions that pairwise statistics miss.

**Differentiability**
   TRAILED's implementation supports gradients, enabling topological objectives as training-time regularizers.

What is ECT?
------------

The **Euler Characteristic Transform** is a topological descriptor that captures shape information through directional filtrations:

1. **Direction sampling**: Choose directions on the unit sphere
2. **Filtration**: Sweep a hyperplane along each direction
3. **Euler characteristic**: Count connected components, holes, and voids at each level
4. **Vectorization**: Concatenate all curves into a fixed-length descriptor

.. mermaid::

   graph LR
      subgraph "Input"
         A["Patient Embeddings"]
      end

      subgraph "Direction Sampling"
         B1["θ₁"]
         B2["θ₂"]
         B3["θₙ"]
      end

      subgraph "Filtration"
         C1["EC curve 1"]
         C2["EC curve 2"]
         C3["EC curve n"]
      end

      subgraph "Output"
         D["Topological Descriptor"]
      end

      A --> B1
      A --> B2
      A --> B3
      B1 --> C1
      B2 --> C2
      B3 --> C3
      C1 --> D
      C2 --> D
      C3 --> D

      style A fill:#f9f9f9,stroke:#999
      style D fill:#DFF0D8,stroke:#3C763D,stroke-width:2px

ECT descriptors have powerful properties:

**Injectivity**
   ECT can distinguish between almost all shapes — it's injective on a dense subset of shapes.

**Stability**
   Small perturbations to input data produce small changes in descriptors.

**Differentiability**
   TRAILED's implementation supports gradients for end-to-end learning.

Roadmap
-------

TRAILED is being developed in phases:

**Current: ECT Foundation**
   Fast, differentiable ECT computation with NumPy, sklearn, and PyTorch integrations. This is the building block for healthcare-specific methods.

**Planned: Density-Aware Descriptors**
   Extensions that fuse topological structure with local density information, addressing limitations of standard ECT for statistical inference.

**Planned: Patient Manifold**
   Methods for constructing and analyzing patient manifolds from longitudinal EHR embeddings, characterizing viable pathways vs. impossible states.

**Planned: Fidelity Metrics**
   Topological fidelity scores for synthetic data validation, designed to correlate with downstream clinical task utility.

Architecture
------------

TRAILED has a layered design:

.. mermaid::

   graph TB
      subgraph "Core"
         A["Direction sampling"]
         B["Filtration computation"]
         C["EC curve calculation"]
      end

      subgraph "Adapters"
         D["NumPy interface"]
         E["pandas interface"]
      end

      subgraph "Plugins (optional)"
         F["sklearn transformer"]
         G["PyTorch layer"]
      end

      A --> B
      B --> C
      C --> D
      C --> E
      D --> F
      D --> G

      style A fill:#D9EDF7,stroke:#31708F,stroke-width:2px
      style B fill:#D9EDF7,stroke:#31708F,stroke-width:2px
      style C fill:#D9EDF7,stroke:#31708F,stroke-width:2px

Key Components
--------------

**Direction Sampling**

TRAILED supports multiple sampling strategies:

- **Uniform**: Random directions on the sphere
- **Stratified**: Evenly distributed directions
- **Custom**: User-defined direction sets

**Resolution Control**

The ``resolution`` parameter controls filtration granularity:

- Higher resolution = more detail, larger descriptors
- Lower resolution = faster computation, coarser features

**Framework Integration**

=============== =====================================================
Framework        Installation
=============== =====================================================
NumPy/pandas     ``pip install trailed`` (included)
sklearn          ``pip install trailed[sklearn]``
PyTorch          ``pip install trailed[torch]``
All              ``pip install trailed[all]``
=============== =====================================================

Use Cases
---------

**Synthetic Data Validation**
   Compare topological structure of real and synthetic EHR cohorts to detect mode collapse and coverage gaps.

**Training Regularization**
   Use differentiable ECT as a loss term to steer generative models away from pathological solutions.

**Patient Trajectory Analysis**
   Characterize clinical pathways and identify anomalous trajectories in topological latent space.

**Representation Learning**
   Extract topological features from longitudinal health records for downstream prediction tasks.

Next Steps
----------

- :ref:`Quickstart <quickstart>` - Compute your first descriptor
- :ref:`User Guide <user_guide>` - Detailed configuration
- :ref:`Integrations <integrations>` - sklearn and PyTorch adapters
